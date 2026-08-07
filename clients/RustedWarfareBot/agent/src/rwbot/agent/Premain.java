package rwbot.agent;

import java.lang.instrument.Instrumentation;

/**
 * Agent entry point. Installs {@link NoOpTransformer}, then forces every
 * targeted class to load so the patch is proven applied before the engine
 * starts, not discovered missing at the first rendered frame.
 *
 * <p>The agent carries no decision logic: it exists to keep a headless engine
 * running and, later, to dispatch orders and serialise state. Anything that
 * chooses is the Python planner's job (wiki: runtime-split-java-agent-python-brain).
 */
public final class Premain {

    private Premain() {
    }

    public static void premain(String argument, Instrumentation instrumentation) {
        AgentOptions options = AgentOptionsParser.parse(argument);

        java.util.Map<String, java.util.Set<String>> targets = Targets.byClass();
        NoOpTransformer transformer = new NoOpTransformer(targets);
        instrumentation.addTransformer(transformer);

        forceLoad(targets.keySet());

        java.util.List<String> unseen = transformer.unseen();
        if (!unseen.isEmpty()) {
            // Hard failure. A silently unpatched engine boots and then dies with
            // the original NullPointerException, which reads as "the fix did not
            // work" rather than "the obfuscated name moved in this build".
            throw new IllegalStateException(
                    "rw-agent: targeted classes were not patched: " + unseen
                            + " -- the pinned build is 1.15 (code 176, build #28);"
                            + " obfuscated names change between releases, so re-derive"
                            + " them against this jar and update Targets.");
        }

        // Synchronous pathfinding: the one patch that touches simulation
        // timing, deliberately (see SyncPathTransformer). Skipped when hosting
        // because a private sim change desyncs against a stock-engine peer;
        // everywhere else it is what makes one seed produce one answer.
        if (!options.hostRequested()) {
            SyncPathTransformer syncPath = new SyncPathTransformer();
            instrumentation.addTransformer(syncPath);
            forceLoad(java.util.Collections.singleton(SyncPathTransformer.PATH_SOLVER));
            if (!syncPath.patched()) {
                throw new IllegalStateException(
                        "rw-agent: path solver was not patched: "
                                + SyncPathTransformer.PATH_SOLVER
                                + " -- the pinned build is 1.15 (code 176, build #28);"
                                + " re-derive the solver's obfuscated name against this jar.");
            }
        }

        // Intent only: the swap itself waits for match start, because a
        // premain swap was measured being overwritten by the holder's own
        // <clinit> (see RandomTap).
        if (options.rngTapRequested()) {
            RandomTap.arm();
        }

        // Before anything else that could draw from it. Seeding after the
        // engine has already made choices would pin only the tail of a run.
        if (options.seedRequested()) {
            EngineRandom.seed(options.randomSeed());
        }

        if (options.discoveryRequested()) {
            startDiscovery(
                    options.discoverAtSeconds(),
                    options.exitAfterDiscovery(),
                    options.inspectFields(),
                    options.findElementsUnder(),
                    options.stateOutPath(),
                    options.typeFlagsPath(),
                    options.aiZones());
        }
        if (options.orderRequested()) {
            startOrderProbe(
                    options.orderMoveAtSeconds(),
                    options.orderMoveBy(),
                    options.orderMoveUnitIndex(),
                    options.buildType());
        }
        if (options.channelRequested()) {
            CommandChannel channel =
                    new CommandChannel(
                            options.channelPort(),
                            options.sampleIntervalMs(),
                            options.lockstepFrames(),
                            options.matchRequested());
            // A requested match replaces the engine's game object, so the
            // channel is opened only once that match exists -- see MatchSetup
            // for what sampling the discarded one cost. The watcher also
            // reseeds and arms the hold on the match's first tick, which is
            // what pins the run to the seed (wiki: policy-determinism).
            if (options.hostRequested()) {
                // Sparring: the bot hosts, a human joins, nothing is held or
                // reseeded — the reproducibility machinery is a desync against
                // a real peer (wiki: multiplayer-portability-invariants).
                MatchSetup.scheduleHost(options.hostMap(), channel);
            } else if (options.matchRequested()) {
                MatchSetup.schedule(
                        options.matchMap(),
                        options.matchOpponents(),
                        options.matchDifficulty(),
                        options.randomSeed(),
                        channel,
                        options.pinDeltaMs(),
                        options.fastForwardFps());
            } else {
                channel.start();
            }
        }
        Log.info("ready; patched " + targets.size() + " class(es)");
    }

    /**
     * Starts a daemon thread that snapshots the engine at each requested time.
     *
     * <p>Daemon so a probe can never hold the JVM open past the game, and off
     * the game thread so a snapshot cannot pace the simulation.
     *
     * @param atSeconds Elapsed times to snapshot at, ascending.
     */
    private static void startDiscovery(
            int[] atSeconds,
            boolean exitAfter,
            String[] inspectFields,
            String findElementsUnder,
            String stateOutPath,
            String typeFlagsPath,
            boolean aiZones) {
        Thread thread =
                new Thread(
                        () -> {
                            runDiscovery(
                                    atSeconds,
                                    inspectFields,
                                    findElementsUnder,
                                    stateOutPath,
                                    typeFlagsPath,
                                    aiZones);
                            if (exitAfter) {
                                Log.info("discovery complete; halting");
                                Runtime.getRuntime().halt(0);
                            }
                        },
                        "rw-agent-discovery");
        thread.setDaemon(true);
        thread.start();
        Log.info("discovery scheduled at " + java.util.Arrays.toString(atSeconds) + "s");
    }

    /** Sleeps to each offset in turn and emits one snapshot per offset. */
    private static void runDiscovery(
            int[] atSeconds,
            String[] inspectFields,
            String findElementsUnder,
            String stateOutPath,
            String typeFlagsPath,
            boolean aiZones) {
        long started = System.nanoTime();
        boolean flagsWritten = false;
        for (int second : atSeconds) {
            long targetNanos = started + second * 1_000_000_000L;
            long remaining = targetNanos - System.nanoTime();
            if (remaining > 0) {
                try {
                    Thread.sleep(remaining / 1_000_000L);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    Log.error("discovery interrupted before t=" + second + "s");
                    return;
                }
            }
            Object engine = EngineHandle.current();
            Log.info(Snapshot.describe(engine, "t=" + second + "s"));
            for (String fieldName : inspectFields) {
                Log.info(Snapshot.describeElements(engine, fieldName));
            }
            if (!findElementsUnder.isEmpty()) {
                Log.info(GraphSearch.findCollections(engine, findElementsUnder, 4, 20000));
            }
            if (!stateOutPath.isEmpty()) {
                writeSample(engine, stateOutPath);
            }
            // Once, not once per offset. The placement flags are fixed for the
            // life of the process -- unit types load with the assets and the
            // mod set is decided at boot -- so a second dump could only ever
            // repeat the first.
            if (!typeFlagsPath.isEmpty() && !flagsWritten) {
                writeTypeFlags(typeFlagsPath);
                flagsWritten = true;
            }
            // Every offset, unlike the placement flags: the whole point is to
            // watch cooldowns and group sizes change, so one dump would answer
            // nothing. Logged, never streamed -- see AiZones for why.
            if (aiZones) {
                Log.info(AiZones.describe(engine));
            }
        }
    }

    /** Seconds after the order at which position is re-sampled to prove movement. */
    private static final int[] ORDER_SAMPLE_OFFSETS = {2, 5, 10};

    /**
     * Issues one real move order and samples the unit's position around it.
     *
     * <p>This is the first time the agent writes to the simulation rather than
     * reading it. The proof obligation is movement, not absence of error: an
     * order that is accepted, queued and silently dropped looks identical to a
     * successful one unless the position is watched afterwards. So the sequence
     * is sample, order, then sample again at increasing offsets.
     *
     * <p>Every engine touch is posted to the game thread. Reads are as unsafe
     * as writes here -- a position sampled mid-tick can be torn -- and the
     * engine already provides the queue for this.
     *
     * @param atSeconds Elapsed time at which to issue the order.
     * @param moveBy World-space offset to send the unit by.
     */
    private static void startOrderProbe(
            int atSeconds, float[] moveBy, int unitIndex, String buildType) {
        Thread thread =
                new Thread(
                        () -> runOrderProbe(atSeconds, moveBy, unitIndex, buildType),
                        "rw-agent-order");
        thread.setDaemon(true);
        thread.start();
        Log.info(
                (buildType.isEmpty() ? "move" : "build " + buildType)
                        + " order scheduled at "
                        + atSeconds
                        + "s for roster["
                        + unitIndex
                        + "], offset ("
                        + moveBy[0]
                        + ", "
                        + moveBy[1]
                        + ")");
    }

    /** Waits for each offset in turn, ordering once and then sampling. */
    private static void runOrderProbe(
            int atSeconds, float[] moveBy, int unitIndex, String buildType) {
        long started = System.nanoTime();
        if (!sleepUntil(started, atSeconds)) {
            return;
        }

        java.util.concurrent.atomic.AtomicReference<Object> ordered =
                new java.util.concurrent.atomic.AtomicReference<Object>();
        Orders.onGameThread(
                () -> {
                    Object engine = EngineHandle.current();
                    Log.info(Mobility.describeOwned(engine));
                    java.util.List<Object> roster = Perception.ownedUnits(engine);
                    if (unitIndex >= roster.size()) {
                        Log.error(
                                "order: roster["
                                        + unitIndex
                                        + "] requested but the player owns "
                                        + roster.size()
                                        + " order-taking entities");
                        return;
                    }
                    Object unit = roster.get(unitIndex);
                    float[] from = Perception.positionOf(unit);
                    float toX = from[0] + moveBy[0];
                    float toY = from[1] + moveBy[1];
                    Log.info("order: subject " + Perception.describe(unit));
                    if (buildType.isEmpty()) {
                        Log.info("order: moving to (" + toX + ", " + toY + ")");
                        Orders.moveTo(engine, unit, toX, toY);
                    } else {
                        Log.info(
                                "order: building "
                                        + buildType
                                        + " at ("
                                        + toX
                                        + ", "
                                        + toY
                                        + ")");
                        Orders.buildAt(engine, unit, buildType, toX, toY);
                    }
                    ordered.set(unit);
                    Log.info("order: issued");
                });

        for (int offset : ORDER_SAMPLE_OFFSETS) {
            if (!sleepUntil(started, atSeconds + offset)) {
                return;
            }
            int elapsed = offset;
            Orders.onGameThread(
                    () -> {
                        Object unit = ordered.get();
                        if (unit == null) {
                            return;
                        }
                        Log.info("order: t+" + elapsed + "s " + Perception.describe(unit));
                        Log.info(Mobility.describeOwned(EngineHandle.current()));
                    });
        }
    }

    /**
     * Sleeps until a given elapsed offset from a start instant.
     *
     * @param startedNanos Reference instant from {@code System.nanoTime}.
     * @param second Offset to wake at.
     * @return True when the offset was reached, false when interrupted.
     */
    private static boolean sleepUntil(long startedNanos, int second) {
        long remaining = startedNanos + second * 1_000_000_000L - System.nanoTime();
        if (remaining <= 0) {
            return true;
        }
        try {
            Thread.sleep(remaining / 1_000_000L);
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("order probe interrupted before t=" + second + "s");
            return false;
        }
    }

    /**
     * Loads each target without initialising it.
     *
     * <p>Initialisation is deliberately skipped: the static initialiser of
     * {@code com.corrodinggames.rts.java.d.a} calls
     * {@code Renderer.get()}, which reaches OpenGL state that does not exist
     * this early. Resolution alone is enough to fire the transformer.
     */
    private static void forceLoad(java.util.Set<String> internalNames) {
        ClassLoader loader = Premain.class.getClassLoader();
        for (String internalName : internalNames) {
            String binaryName = internalName.replace('/', '.');
            try {
                Class.forName(binaryName, false, loader);
            } catch (ClassNotFoundException e) {
                throw new IllegalStateException(
                        "rw-agent: target class not found on the classpath: " + binaryName, e);
            }
        }
    }

    /**
     * Appends one NDJSON world sample to the stream file.
     *
     * <p>The read runs on the game thread and the write does not: file I/O on
     * the simulation thread would pace the game by disk latency, while the
     * sample itself must be coherent. So the sample is built on the game thread
     * and handed back to be written here.
     *
     * @param engine The live engine instance.
     * @param path Absolute path to append to.
     */
    private static void writeSample(Object engine, String path) {
        String rendered = renderOnGameThread(() -> StateStream.sample(engine), "state sample");
        if (rendered == null) {
            Log.error("state sample was not produced");
            return;
        }
        try (java.io.Writer writer =
                new java.io.OutputStreamWriter(
                        new java.io.FileOutputStream(path, true),
                        java.nio.charset.StandardCharsets.UTF_8)) {
            writer.write(rendered);
        } catch (java.io.IOException e) {
            throw new IllegalStateException("rw-agent: cannot append state to " + path, e);
        }
        Log.info("state sample appended to " + path);
    }

    /**
     * Writes the unit-type placement flags.
     *
     * <p>Truncating rather than appending, which is the opposite of the state
     * stream and deliberate: the stream is a growing corpus of observations,
     * while this is one complete answer to a question with one answer.
     * Appending would produce a file with the same types listed twice.
     *
     * @param path Absolute path to write.
     */
    private static void writeTypeFlags(String path) {
        // Both kinds ride in one file, taken in one pass over one registry.
        // They answer different questions about the same types -- where each
        // may stand, and what each can make -- and two files could be
        // regenerated against different game builds and silently disagree.
        String rendered = renderOnGameThread(TypeFlags::dump, "type flags");
        String edges = renderOnGameThread(BuildTree::dump, "build tree");
        if (rendered == null || edges == null) {
            Log.error("type flags were not produced");
            return;
        }
        rendered = rendered + edges;
        try (java.io.Writer writer =
                new java.io.OutputStreamWriter(
                        new java.io.FileOutputStream(path, false),
                        java.nio.charset.StandardCharsets.UTF_8)) {
            writer.write(rendered);
        } catch (java.io.IOException e) {
            throw new IllegalStateException("rw-agent: cannot write type flags to " + path, e);
        }
        Log.info("type flags written to " + path);
    }

    /**
     * Runs a read against the live simulation and returns what it rendered.
     *
     * <p>{@code onGameThread} enqueues and returns; it does not run the task.
     * Reading the result straight after would always see nothing, so the render
     * is awaited explicitly. The wait is bounded and fails loudly rather than
     * hanging a probe against a stalled game.
     *
     * @param render The read, which runs on the game thread.
     * @param what Name of the read, for the failure message.
     * @return What it rendered, or null when it produced nothing.
     */
    private static String renderOnGameThread(
            java.util.function.Supplier<String> render, String what) {
        final java.util.concurrent.atomic.AtomicReference<String> rendered =
                new java.util.concurrent.atomic.AtomicReference<String>();
        final java.util.concurrent.atomic.AtomicReference<RuntimeException> failure =
                new java.util.concurrent.atomic.AtomicReference<RuntimeException>();
        final java.util.concurrent.CountDownLatch done =
                new java.util.concurrent.CountDownLatch(1);

        Orders.onGameThread(
                () -> {
                    try {
                        rendered.set(render.get());
                    } catch (RuntimeException e) {
                        failure.set(e);
                    } finally {
                        done.countDown();
                    }
                });

        boolean completed;
        try {
            completed = done.await(10, java.util.concurrent.TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("rw-agent: interrupted awaiting a " + what, e);
        }
        if (!completed) {
            throw new IllegalStateException(
                    "rw-agent: the game thread did not render a " + what + " within 10s");
        }
        // Carried across the thread boundary rather than swallowed there, so a
        // failure surfaces with its original stack on the probe thread.
        RuntimeException thrown = failure.get();
        if (thrown != null) {
            throw thrown;
        }
        return rendered.get();
    }

}
