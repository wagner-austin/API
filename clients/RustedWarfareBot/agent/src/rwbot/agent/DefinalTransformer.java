package rwbot.agent;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * Clears {@code ACC_FINAL} on the engine generator holder's field as the
 * class loads, so the match-start generator swap is visible to every
 * caller -- the JIT constant-fold seam {@link Definal} documents.
 *
 * <p>Unconditional, unlike the diagnostic transformers: this is what makes
 * certified play's seeded swap ({@link SplitRandom}) actually reach a draw
 * helper the menu demo already compiled, so it belongs to every regime,
 * hosting included -- the patch is simulation-neutral until the swap,
 * and the swap keeps its own containment.
 *
 * <p>Same accounting contract as the other transformers: {@link Premain}
 * fails loudly when the holder never patched.
 */
final class DefinalTransformer implements ClassFileTransformer {

    private volatile boolean patched;

    @Override
    public byte[] transform(
            ClassLoader loader,
            String className,
            Class<?> classBeingRedefined,
            ProtectionDomain protectionDomain,
            byte[] classfileBuffer) {

        if (!Targets.GENERATOR_HOLDER.equals(className)) {
            return null;
        }

        // Same containment as the other transformers: a throw from
        // transform() is swallowed by the JVM and the original bytes load
        // unchanged, so the failure is reported here and turned into a hard
        // stop by Premain's accounting.
        byte[] result;
        try {
            result = Definal.strip(classfileBuffer, Targets.generatorDefinals());
        } catch (RuntimeException e) {
            Log.error("failed to clear final on " + className + ": " + e);
            return null;
        } catch (ClassFormatError e) {
            Log.error("failed to clear final on " + className + ": " + e);
            return null;
        }

        patched = true;
        Log.info(
                "cleared final on " + className + " field(s) "
                        + Targets.generatorDefinals().keySet()
                        + " -- generator swaps now reach compiled callers");
        return result;
    }

    /** Whether the holder class was seen and patched. */
    boolean patched() {
        return patched;
    }
}
