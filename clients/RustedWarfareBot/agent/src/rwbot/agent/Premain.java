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
        AgentOptions options = AgentOptions.parse(argument);

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

        if (options.discoveryRequested()) {
            startDiscovery(
                    options.discoverAtSeconds(),
                    options.exitAfterDiscovery(),
                    options.inspectFields(),
                    options.findElementsUnder(),
                    options.stateOutPath());
        }
        if (options.orderRequested()) {
            startOrderProbe(
                    options.orderMoveAtSeconds(),
                    options.orderMoveBy(),
                    options.orderMoveUnitIndex(),
                    options.buildType());
        }
        if (options.channelRequested()) {
            new CommandChannel(options.channelPort(), options.sampleIntervalMs()).start();
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
            String stateOutPath) {
        Thread thread =
                new Thread(
                        () -> {
                            runDiscovery(atSeconds, inspectFields, findElementsUnder, stateOutPath);
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
            String stateOutPath) {
        long started = System.nanoTime();
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
            Log.info(Discovery.describe(engine, "t=" + second + "s"));
            for (String fieldName : inspectFields) {
                Log.info(Discovery.describeElements(engine, fieldName));
            }
            if (!findElementsUnder.isEmpty()) {
                Log.info(Discovery.findCollections(engine, findElementsUnder, 4, 20000));
            }
            if (!stateOutPath.isEmpty()) {
                writeSample(engine, stateOutPath);
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
                    Log.info(Perception.describeOwned(engine));
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
                        Log.info(Perception.describeOwned(EngineHandle.current()));
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
        // onGameThread enqueues and returns; it does not run the task. Reading
        // the result straight after would always see nothing, so the render is
        // awaited explicitly. The wait is bounded and fails loudly rather than
        // hanging a probe against a stalled game.
        final java.util.concurrent.atomic.AtomicReference<String> rendered =
                new java.util.concurrent.atomic.AtomicReference<String>();
        final java.util.concurrent.atomic.AtomicReference<RuntimeException> failure =
                new java.util.concurrent.atomic.AtomicReference<RuntimeException>();
        final java.util.concurrent.CountDownLatch done =
                new java.util.concurrent.CountDownLatch(1);

        Orders.onGameThread(
                () -> {
                    try {
                        rendered.set(StateStream.sample(engine));
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
            throw new IllegalStateException("rw-agent: interrupted awaiting a state sample", e);
        }
        if (!completed) {
            throw new IllegalStateException(
                    "rw-agent: the game thread did not render a state sample within 10s");
        }
        // Carried across the thread boundary rather than swallowed there, so a
        // failure surfaces with its original stack on the probe thread.
        RuntimeException thrown = failure.get();
        if (thrown != null) {
            throw thrown;
        }
        if (rendered.get() == null) {
            Log.error("state sample was not produced");
            return;
        }
        try (java.io.Writer writer =
                new java.io.OutputStreamWriter(
                        new java.io.FileOutputStream(path, true),
                        java.nio.charset.StandardCharsets.UTF_8)) {
            writer.write(rendered.get());
        } catch (java.io.IOException e) {
            throw new IllegalStateException("rw-agent: cannot append state to " + path, e);
        }
        Log.info("state sample appended to " + path);
    }
}
