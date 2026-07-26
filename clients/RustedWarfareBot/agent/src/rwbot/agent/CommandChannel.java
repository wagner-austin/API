package rwbot.agent;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The planner's connection to the running game.
 *
 * <p>One loopback socket carries both directions as newline-delimited JSON:
 * world samples out, orders in. The agent listens and the planner connects,
 * because the game is the long-lived process -- a planner can attach, crash,
 * and reattach without disturbing the match.
 *
 * <p><b>Backpressure is the design constraint.</b> Samples are produced on the
 * game thread and written by a separate thread through a small bounded queue
 * that drops its oldest entry when full. A planner that stops reading must
 * never be able to stall the simulation, and blocking a socket write on the
 * game thread is exactly how that would happen. Dropping stale world state is
 * the right loss: the next sample supersedes it anyway.
 *
 * <p>Orders arrive on the reader thread and are dispatched through
 * {@link Orders#onGameThread}, so nothing here touches the simulation directly
 * (wiki: issuing-orders).
 */
final class CommandChannel {

    /** Samples buffered before the oldest is dropped. Small on purpose. */
    private static final int OUTBOX_DEPTH = 4;

    /**
     * How long the simulation may wait for the planner before giving up on
     * lockstep for that step.
     *
     * <p>Bounded because the alternative is a hung game. A planner that has
     * died, or is simply thinking for too long, must not be able to freeze the
     * engine indefinitely -- so the wait expires, says so, and the game runs
     * on. That makes the run non-reproducible from that point, which is why it
     * is logged as an error rather than a note.
     */
    private static final long ACK_TIMEOUT_MS = 15_000L;

    private final int port;
    private final int sampleIntervalMs;
    private final int lockstepFrames;
    private final java.util.concurrent.SynchronousQueue<Boolean> acks =
            new java.util.concurrent.SynchronousQueue<Boolean>();
    private int nextSampleFrame = -1;
    private volatile boolean armed = false;
    private final ArrayBlockingQueue<String> outbox =
            new ArrayBlockingQueue<String>(OUTBOX_DEPTH);
    private final AtomicBoolean connected = new AtomicBoolean(false);

    CommandChannel(int port, int sampleIntervalMs, int lockstepFrames) {
        this.port = port;
        this.sampleIntervalMs = sampleIntervalMs;
        this.lockstepFrames = lockstepFrames;
    }

    /**
     * Starts listening and serving, on daemon threads.
     *
     * <p>Daemon so a planner that never connects cannot hold the JVM open past
     * the game.
     */
    void start() {
        Thread listener = new Thread(this::serve, "rw-agent-channel");
        listener.setDaemon(true);
        listener.start();

        Thread sampler = new Thread(this::sampleLoop, "rw-agent-sampler");
        sampler.setDaemon(true);
        sampler.start();

        Log.info(
                "channel listening on 127.0.0.1:"
                        + port
                        + ", sampling every "
                        + sampleIntervalMs
                        + "ms");
    }

    /** Accepts one planner at a time, serving until it disconnects. */
    private void serve() {
        ServerSocket server;
        try {
            server = new ServerSocket(port, 1, InetAddress.getLoopbackAddress());
        } catch (IOException e) {
            // Fatal and worth being loud about: without the channel the agent
            // is a no-op patcher, and a silently unbound port would look like
            // a planner bug rather than an agent one.
            Log.error("channel could not bind 127.0.0.1:" + port + ": " + e);
            return;
        }
        while (true) {
            Socket socket;
            try {
                socket = server.accept();
            } catch (IOException e) {
                Log.error("channel accept failed: " + e);
                return;
            }
            Log.info("channel: planner connected");
            outbox.clear();
            connected.set(true);
            armed = false;
            nextSampleFrame = -1;
            serveOne(socket);
            connected.set(false);
            // Release a step that is still waiting on the planner that just
            // left, rather than making the simulation serve out the full ack
            // timeout for an answer that can no longer come. A readiness probe
            // that opens the port and closes it is the ordinary case.
            acks.offer(Boolean.TRUE);
            Log.info("channel: planner disconnected");
        }
    }

    /** Runs one connection: a writer thread out, this thread reading in. */
    private void serveOne(Socket socket) {
        Thread writer = new Thread(() -> writeLoop(socket), "rw-agent-channel-out");
        writer.setDaemon(true);
        writer.start();
        readLoop(socket);
        try {
            socket.close();
        } catch (IOException e) {
            Log.error("channel: closing socket failed: " + e);
        }
        // The writer is retired before the next planner is served, and this is
        // not tidiness. It blocks taking from the shared outbox, so a writer
        // left over from a closed connection goes on competing for samples --
        // it took the first sample of the following connection, wrote it to a
        // dead socket, and left the real planner waiting for state that had
        // already been consumed. Under lockstep that showed up as the
        // simulation stalling for the whole ack timeout on every run.
        connected.set(false);
        writer.interrupt();
        try {
            writer.join(1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Reads one order per line and dispatches it on the game thread. */
    private void readLoop(Socket socket) {
        BufferedReader reader;
        try {
            reader =
                    new BufferedReader(
                            new InputStreamReader(
                                    socket.getInputStream(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            Log.error("channel: could not read from the planner: " + e);
            return;
        }
        while (true) {
            String line;
            try {
                line = reader.readLine();
            } catch (IOException e) {
                Log.error("channel: read failed: " + e);
                return;
            }
            if (line == null) {
                return;
            }
            if (line.trim().isEmpty()) {
                continue;
            }
            accept(line);
        }
    }

    /**
     * Parses and dispatches one order.
     *
     * <p>A malformed order is reported and the connection continues. That is
     * not best-effort softening of a failure: the planner is a separate process
     * and one bad line is its bug, not grounds for dropping a live match. The
     * rejection is logged with the offending line so it cannot pass unnoticed.
     */
    private void accept(String line) {
        CommandRecord command;
        try {
            command = CommandRecord.parse(line);
        } catch (IllegalArgumentException e) {
            Log.error("channel: rejected order: " + e.getMessage());
            return;
        }
        if (command.kind() == CommandRecord.Kind.ACK) {
            // Released here rather than on the game thread: the game thread is
            // the one being held, so posting the release into its own queue
            // would deadlock it behind itself.
            ack();
            return;
        }
        Orders.onGameThread(() -> apply(command));
    }

    /** Applies one parsed order. Runs on the game thread. */
    private void apply(CommandRecord command) {
        Object engine = EngineHandle.current();
        Object unit = Perception.findOwnedById(engine, command.unitId());
        if (unit == null) {
            Log.error(
                    "channel: no owned unit with id "
                            + command.unitId()
                            + "; it may have died since the sample");
            return;
        }
        if (command.kind() == CommandRecord.Kind.MOVE) {
            Orders.moveTo(engine, unit, command.x(), command.y());
            Log.info(
                    "channel: move "
                            + command.unitId()
                            + " -> ("
                            + command.x()
                            + ", "
                            + command.y()
                            + ")");
            return;
        }
        if (command.kind() == CommandRecord.Kind.ATTACK) {
            // The target is looked up among what is visible rather than what is
            // owned, and it is looked up now rather than trusted from the
            // sample: a target can die or slip back into fog between the
            // planner deciding and the order arriving, and an attack on a
            // stale identity is a command the engine would drop silently.
            Object target = Perception.findVisibleById(engine, command.targetId());
            if (target == null) {
                Log.error(
                        "channel: no visible unit with id "
                                + command.targetId()
                                + "; the target died or left sight since the sample");
                return;
            }
            Orders.attack(engine, unit, target);
            Log.info("channel: attack " + command.targetId() + " by " + command.unitId());
            return;
        }
        if (command.kind() == CommandRecord.Kind.PRODUCE) {
            Orders.produce(engine, unit, command.buildType());
            Log.info(
                    "channel: produce "
                            + command.buildType()
                            + " by "
                            + command.unitId());
            return;
        }
        Orders.buildAt(engine, unit, command.buildType(), command.x(), command.y());
        Log.info(
                "channel: build "
                        + command.buildType()
                        + " by "
                        + command.unitId()
                        + " at ("
                        + command.x()
                        + ", "
                        + command.y()
                        + ")");
    }

    /** Drains sampled world state to the planner. */
    private void writeLoop(Socket socket) {
        Writer out;
        try {
            out =
                    new OutputStreamWriter(
                            socket.getOutputStream(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            Log.error("channel: could not write to the planner: " + e);
            return;
        }
        while (connected.get()) {
            String sample;
            try {
                sample = outbox.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            try {
                out.write(sample);
                out.flush();
            } catch (IOException e) {
                Log.error("channel: write failed, dropping the planner: " + e);
                return;
            }
        }
    }

    /**
     * Records that the planner has finished with the current sample.
     *
     * <p>Called from the reader thread. The handoff is a rendezvous rather
     * than a flag so a late ack from a previous step cannot release the next
     * one: if nothing is waiting, the ack is dropped, which is the correct
     * reading of an ack nobody asked for.
     */
    void ack() {
        acks.offer(Boolean.TRUE);
    }

    /**
     * Runs one lockstep step on the game thread, then re-arms itself.
     *
     * <p>The engine drains this queue from its own update, so a runnable that
     * posts itself again runs once per tick -- an every-tick hook without
     * touching a single byte of engine code.
     *
     * <p>At each frame boundary the sample is published and the simulation is
     * held until the planner answers. Holding the game thread is the whole
     * point: it is what stops the planner's think time from deciding which
     * frame its orders land on, and that coupling was the largest remaining
     * source of run-to-run divergence once the engine's randomness was pinned
     * (wiki: policy-determinism).
     */
    private void lockstepTick() {
        Object engine = EngineHandle.current();
        int frame = EngineAccess.readIntField(engine, StateStream.FRAME_FIELD);
        if (nextSampleFrame < 0) {
            nextSampleFrame = frame;
        }
        if (connected.get() && frame >= nextSampleFrame) {
            offer(StateStream.sample(engine));
            nextSampleFrame = frame + lockstepFrames;
            awaitAck();
        }
        if (connected.get()) {
            Orders.onGameThread(this::lockstepTick);
        }
    }

    /**
     * Blocks the simulation until the planner acks, the planner leaves, or the
     * bound expires.
     *
     * <p>Waits in slices and re-tests rather than parking once on a handoff.
     * A planner that disconnects mid-step cannot be relied on to release a
     * waiter -- the rendezvous only completes if a consumer is already parked,
     * and a readiness probe that opens the port and closes it loses that race
     * routinely. Re-testing the connection makes departure a condition rather
     * than a message.
     */
    private void awaitAck() {
        long deadline = System.nanoTime() + ACK_TIMEOUT_MS * 1_000_000L;
        while (System.nanoTime() < deadline) {
            if (!connected.get()) {
                return;
            }
            try {
                if (acks.poll(200L, java.util.concurrent.TimeUnit.MILLISECONDS) != null) {
                    return;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
        Log.error(
                "channel: no ack within " + ACK_TIMEOUT_MS
                        + "ms; running on unlocked, so this run is no longer reproducible");
    }

    /** Samples the world on the game thread at a fixed cadence. */
    private void sampleLoop() {
        while (true) {
            try {
                Thread.sleep(sampleIntervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            if (!connected.get()) {
                continue;
            }
            if (lockstepFrames > 0) {
                // In lockstep the tick hook owns sampling entirely; a second
                // source would publish samples the planner never acked.
                //
                // Arming waits for the engine to be able to accept work rather
                // than for a connection. The socket binds before the game has
                // loaded a map, and a readiness probe can connect in that
                // window -- arming on connect threw on the channel thread and
                // took the channel down with it. The condition is tested
                // directly, so a genuine binding failure still throws instead
                // of being mistaken for "not started yet".
                if (!armed && Orders.gameThreadReady()) {
                    Orders.onGameThread(this::lockstepTick);
                    armed = true;
                    Log.info(
                            "channel: lockstep every " + lockstepFrames
                                    + " frame(s); the simulation waits for the planner");
                }
                continue;
            }
            Orders.onGameThread(
                    () -> {
                        String sample = StateStream.sample(EngineHandle.current());
                        offer(sample);
                    });
        }
    }

    /**
     * Queues one sample, dropping the oldest when the planner is behind.
     *
     * <p>Called on the game thread, so it must never block. Visible for the
     * self-test, which asserts the drop rather than trusting the comment.
     *
     * @param sample The rendered sample.
     * @return True when the sample was queued without a drop.
     */
    boolean offer(String sample) {
        if (outbox.offer(sample)) {
            return true;
        }
        outbox.poll();
        outbox.offer(sample);
        return false;
    }

    /** Buffered sample count. Visible for the self-test. */
    int queued() {
        return outbox.size();
    }

    /** Marks the channel connected. Visible for the self-test. */
    void setConnectedForTest(boolean value) {
        connected.set(value);
    }
}
