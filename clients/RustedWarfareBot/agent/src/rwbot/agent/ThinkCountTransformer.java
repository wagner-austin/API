package rwbot.agent;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * Injects the think-entry counters as the AI classes load.
 *
 * <p>The wall-decoupling program's first instrument: every prior probe
 * counted draws or read state, both of which sat identical across twins
 * whose worlds forked, and only the INVOCATION count of the think passes
 * separates "ran and found nothing" from "never ran" ({@link ThinkCount},
 * wiki: policy-determinism). {@link EntryCounts#prepend}
 * prepends a four-byte {@code invokestatic}+{@code nop} at each entry in
 * {@link Targets#thinkCounters()}; the methods' own bytes are otherwise
 * untouched, so nothing about WHAT a think does changes -- only that its
 * running is now on the record. The edit arithmetic lives in
 * {@link EntryCounts}, split beside {@link Bytecode} at the module ceiling.
 *
 * <p>Diagnostic only, armed with the draw tap like {@link AiCadence}: a
 * measurement run states its regime, and certified play keeps original
 * bytes. Same accounting contract as {@link SwayRouteTransformer}:
 * {@link Premain} fails loudly when a targeted class never patched.
 */
final class ThinkCountTransformer implements ClassFileTransformer {

    private final java.util.Map<String, java.util.LinkedHashMap<String, String>> targets =
            Targets.thinkCounters();
    private final java.util.Map<String, java.util.LinkedHashMap<String, String>> scans =
            Targets.thinkScans();
    private final java.util.Set<String> patched =
            java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<String>());

    @Override
    public byte[] transform(
            ClassLoader loader,
            String className,
            Class<?> classBeingRedefined,
            ProtectionDomain protectionDomain,
            byte[] classfileBuffer) {

        if (className == null) {
            return null;
        }
        java.util.LinkedHashMap<String, String> counters = targets.get(className);
        if (counters == null) {
            return null;
        }
        java.util.LinkedHashMap<String, String> receiverHooks = scans.get(className);
        if (receiverHooks == null) {
            receiverHooks = new java.util.LinkedHashMap<String, String>();
        }

        // Same containment as the other transformers: a throw from
        // transform() is swallowed by the JVM and the original bytes load
        // unchanged, so the failure is reported here and turned into a hard
        // stop by Premain's accounting.
        byte[] result;
        try {
            result =
                    EntryCounts.prepend(
                            classfileBuffer, counters, receiverHooks, Targets.THINK_COUNT_OWNER);
        } catch (RuntimeException e) {
            Log.error("failed to count think entries in " + className + ": " + e);
            return null;
        } catch (ClassFormatError e) {
            Log.error("failed to count think entries in " + className + ": " + e);
            return null;
        }

        patched.add(className);
        Log.info("counting think entries in " + className);
        return result;
    }

    /** Targeted classes that were never patched, in declaration order. */
    java.util.List<String> unseen() {
        java.util.List<String> missing = new java.util.ArrayList<String>();
        for (String className : targets.keySet()) {
            if (!patched.contains(className)) {
                missing.add(className);
            }
        }
        return missing;
    }
}
