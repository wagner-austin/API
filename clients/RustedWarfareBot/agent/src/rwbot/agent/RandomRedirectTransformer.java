package rwbot.agent;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * Applies {@link RandomRedirect} to the AI class as it loads -- the
 * seam-2 pin, active in every non-hosting regime because certified play
 * is what needs a deterministic turret choice, exactly as
 * {@link DefinalTransformer} is what makes the generator swap real.
 *
 * <p>Same accounting contract as the other transformers: {@link Premain}
 * fails loudly when the AI class never patched.
 */
final class RandomRedirectTransformer implements ClassFileTransformer {

    static final String AI_CLASS = "com/corrodinggames/rts/game/a/a";

    private volatile boolean patched;

    @Override
    public byte[] transform(
            ClassLoader loader,
            String className,
            Class<?> classBeingRedefined,
            ProtectionDomain protectionDomain,
            byte[] classfileBuffer) {

        if (!AI_CLASS.equals(className)) {
            return null;
        }

        // Same containment as the other transformers: a throw from
        // transform() is swallowed by the JVM and the original bytes load
        // unchanged, so the failure is reported here and turned into a hard
        // stop by Premain's accounting.
        byte[] result;
        try {
            result = RandomRedirect.redirect(classfileBuffer);
        } catch (RuntimeException e) {
            Log.error("failed to redirect new-Random in " + className + ": " + e);
            return null;
        } catch (ClassFormatError e) {
            Log.error("failed to redirect new-Random in " + className + ": " + e);
            return null;
        }

        patched = true;
        Log.info(
                "redirected the AI's inline new-Random to the seeded choice stream in "
                        + className);
        return result;
    }

    /** Whether the AI class was seen and patched. */
    boolean patched() {
        return patched;
    }
}
