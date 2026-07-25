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
                    options.findElementsUnder());
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
            String findElementsUnder) {
        Thread thread =
                new Thread(
                        () -> {
                            runDiscovery(atSeconds, inspectFields, findElementsUnder);
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
            int[] atSeconds, String[] inspectFields, String findElementsUnder) {
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
}
