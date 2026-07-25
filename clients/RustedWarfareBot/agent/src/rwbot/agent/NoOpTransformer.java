package rwbot.agent;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * Replaces the bodies of {@link Targets} methods as their classes are loaded.
 *
 * <p>Records which targeted classes were actually seen so {@link Premain} can
 * fail loudly on a build whose obfuscated names have moved, rather than let the
 * engine boot unpatched and crash later with the original stack trace.
 */
final class NoOpTransformer implements ClassFileTransformer {

    private final java.util.Map<String, java.util.Set<String>> targets;
    private final java.util.Set<String> patched =
            java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<String>());

    NoOpTransformer(java.util.Map<String, java.util.Set<String>> targets) {
        this.targets = targets;
    }

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
        java.util.Set<String> methods = targets.get(className);
        if (methods == null) {
            return null;
        }

        // A throw from transform() is swallowed by the JVM and the original
        // bytes load unchanged, so a failure here would be invisible at exactly
        // the moment it matters. Catch, report, and let Premain's verification
        // turn it into a hard failure.
        byte[] result;
        try {
            result = ClassFilePatcher.noOpMethods(classfileBuffer, methods);
        } catch (RuntimeException e) {
            Log.error("failed to patch " + className + ": " + e);
            return null;
        } catch (ClassFormatError e) {
            Log.error("failed to patch " + className + ": " + e);
            return null;
        }

        if (result == null) {
            Log.error("no method matched in " + className + "; wanted " + methods);
            return null;
        }

        patched.add(className);
        Log.info("patched " + className + " " + methods);
        return result;
    }

    /** Targeted classes that were never loaded, in declaration order. */
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
