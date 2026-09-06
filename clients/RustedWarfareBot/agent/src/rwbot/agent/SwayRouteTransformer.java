package rwbot.agent;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * Rewires the sway updaters' draws to the side stream as their classes load.
 *
 * <p>The per-call-site half of the draw-routing story: {@link SplitRandom}
 * routes by tick phase, which serves these in-tick draws to the simulation
 * stream -- correctly by its rule, and ruinously in effect, because the
 * updaters run on an engagement-paced schedule byte-identical twins do not
 * share (wiki log 2026-09-06). {@link ClassFilePatcher#retargetStaticInvokes}
 * moves exactly the {@code f.d(FF)F} invokes inside
 * {@link Targets#swayRewires} to {@link SideDraw#d}; the methods themselves
 * keep running, so the {@code aN} flag the opponent AI reads stays
 * maintained -- the reason a no-op is not lawful here.
 *
 * <p>Same accounting contract as {@link NoOpTransformer}: {@link Premain}
 * fails loudly when a targeted class never patched, because a silently
 * unrouted build would measure as the very noise this exists to remove.
 */
final class SwayRouteTransformer implements ClassFileTransformer {

    private final java.util.Map<String, java.util.Set<String>> targets = Targets.swayRewires();
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
        java.util.Set<String> methods = targets.get(className);
        if (methods == null) {
            return null;
        }

        // Same containment as NoOpTransformer: a throw from transform() is
        // swallowed by the JVM and the original bytes load unchanged, so the
        // failure is reported here and turned into a hard stop by Premain.
        byte[] result;
        try {
            result =
                    ClassFilePatcher.retargetStaticInvokes(
                            classfileBuffer,
                            methods,
                            Targets.SWAY_DRAW_OWNER,
                            Targets.SWAY_DRAW_NAME,
                            Targets.SWAY_DRAW_DESCRIPTOR,
                            Targets.SWAY_DRAW_TARGET);
        } catch (RuntimeException e) {
            Log.error("failed to rewire " + className + ": " + e);
            return null;
        } catch (ClassFormatError e) {
            Log.error("failed to rewire " + className + ": " + e);
            return null;
        }

        if (result == null) {
            Log.error("no sway draw to rewire in " + className + "; wanted " + methods);
            return null;
        }

        patched.add(className);
        Log.info("rewired sway draws in " + className + " to the side stream");
        return result;
    }

    /** Targeted classes that were never rewired, in declaration order. */
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
