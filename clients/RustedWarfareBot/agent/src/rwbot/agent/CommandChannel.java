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

    private final int port;
    private final int sampleIntervalMs;
    private final ArrayBlockingQueue<String> outbox =
            new ArrayBlockingQueue<String>(OUTBOX_DEPTH);
    private final AtomicBoolean connected = new AtomicBoolean(false);

    CommandChannel(int port, int sampleIntervalMs) {
        this.port = port;
        this.sampleIntervalMs = sampleIntervalMs;
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
            serveOne(socket);
            connected.set(false);
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
        Orders.onGameThread(() -> apply(command));
    }

    /** Applies one parsed order. Runs on the game thread. */
    private void apply(CommandRecord command) {
        Object engine = EngineHandle.current();
        Object unit = Orders.findOwnedById(engine, command.unitId());
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
