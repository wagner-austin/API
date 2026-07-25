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
        Log.info("ready; patched " + targets.size() + " class(es)");
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
