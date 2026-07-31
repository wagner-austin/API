package rwbot.agent;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * Makes pathfinding synchronous, which makes the simulation deterministic.
 *
 * <p>The engine's one nondeterministic mechanism, proven by bisection (wiki:
 * policy-determinism): the path solver is a dedicated thread that sleeps on a
 * monitor. A move order queues a job and {@code a()} notifies; {@code b()}
 * computes and delivers <em>on whatever tick the OS schedules the worker</em>.
 * The unit stands still until its path arrives, so two identical runs diverge
 * by exactly one walk-start tick per order -- 0.26 world units by the first
 * lockstep window, compounding from there.
 *
 * <p>The patch closes the race at its source rather than tightening it:
 * {@code a()} is rewritten to invoke {@code b()} inline, so the path computes
 * on the requesting thread and delivery is deterministic by construction, and
 * {@code c()} -- the thread starter -- is no-opped, so the worker never exists
 * and no spurious wakeup can ever double-run a job. {@code b()} already
 * synchronizes its delivery on the engine's own monitor, and Java monitors are
 * reentrant, so a caller holding that lock stays safe.
 *
 * <p><b>This deliberately alters simulation timing</b>, unlike {@link Targets}'
 * render-path no-ops -- altering it is the point. That is why {@link Premain}
 * skips this transformer when hosting a real peer: a private sim change is a
 * desync against a client running the stock engine (wiki:
 * multiplayer-portability-invariants).
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28)</b>, same regime
 * as {@link Targets}: {@link Premain} fails loudly if the class is never seen.
 */
final class SyncPathTransformer implements ClassFileTransformer {

    /** The path solver, {@code PathSolver-N} threads' Runnable. */
    static final String PATH_SOLVER = "com/corrodinggames/rts/gameFramework/k/o";

    private volatile boolean patched;

    @Override
    public byte[] transform(
            ClassLoader loader,
            String className,
            Class<?> classBeingRedefined,
            ProtectionDomain protectionDomain,
            byte[] classfileBuffer) {

        if (!PATH_SOLVER.equals(className)) {
            return null;
        }

        // Same containment as NoOpTransformer: a throw from transform() is
        // swallowed by the JVM and the original bytes load unchanged, so the
        // failure is reported here and turned into a hard stop by Premain.
        byte[] result;
        try {
            result = patchSolver(classfileBuffer);
        } catch (RuntimeException e) {
            Log.error("failed to patch " + className + ": " + e);
            return null;
        } catch (ClassFormatError e) {
            Log.error("failed to patch " + className + ": " + e);
            return null;
        }

        patched = true;
        Log.info("patched " + className + " [a()V -> this.b()V inline, c()V no-op]");
        return result;
    }

    /**
     * Applies both edits to the solver's class file.
     *
     * <p>Two passes over the bytes rather than one combined edit, because the
     * two edits are different capabilities of the patcher and each pass is
     * already position-safe on its own. The order does not matter; the methods
     * are disjoint.
     *
     * @throws ClassFormatError when either target method is missing, which
     *     means the pinned obfuscated names moved under this build.
     */
    static byte[] patchSolver(byte[] classFile) {
        byte[] noThread =
                ClassFilePatcher.noOpMethods(classFile, java.util.Collections.singleton("c()V"));
        if (noThread == null) {
            throw new ClassFormatError("no c()V in " + PATH_SOLVER);
        }
        byte[] result = ClassFilePatcher.delegateToSelf(noThread, "a()V", "b", "()V");
        if (result == null) {
            throw new ClassFormatError("no a()V in " + PATH_SOLVER);
        }
        return result;
    }

    /** Whether the solver class was seen and successfully rewritten. */
    boolean patched() {
        return patched;
    }
}
