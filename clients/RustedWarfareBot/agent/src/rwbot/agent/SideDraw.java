package rwbot.agent;

import java.util.Random;

/**
 * The side-stream target for rewired engine draw sites.
 *
 * <p>The per-call-site routing fix [[policy-determinism]] carried as "not
 * done" since 2026-08-31, built for the sway updaters first: three unit
 * classes re-randomize render-only sway marks from INSIDE the simulation
 * tick, so the phase-routed split ({@link SplitRandom}) correctly but
 * ruinously serves them from the sim stream -- and their call rate rides
 * engagement timing, which is exactly what byte-identical twins disagree on
 * (wiki log 2026-09-06, findings #4/#5). The updaters cannot be no-opped:
 * the same method maintains the {@code aN} has-target flag the opponent AI
 * reads ({@code game/a/a.java:1586}). So the method keeps running and only
 * its draws move: {@link ClassFilePatcher#retargetStaticInvokes} rewrites
 * their {@code f.d(FF)F} invokes to {@link #d(float, float)} at class load.
 *
 * <p>Public, because the callers are engine classes in another package;
 * resolvable from the engine's classloader because the agent jar is on the
 * system path every classloader delegates to.
 *
 * <p>Mirrors the engine helper it replaces byte-for-byte in semantics --
 * {@code a.nextFloat() * (hi - lo) + lo}, uniform in [lo, hi) -- but from a
 * salted stream of its own, reseeded at match start beside the other three
 * ({@link EngineRandom#seed}). Determinism of THIS stream is not the point
 * (its values land in render-only marks); what matters is that its draws no
 * longer advance the simulation's stream on an engagement-paced schedule.
 */
public final class SideDraw {

    /** Offsets the seed so this stream never correlates with the sim's. */
    private static final long SWAY_SALT = 0x5AA75A175EEDL;

    /** The stream; replaced whole at each reseed, never reused across matches. */
    private static volatile Random stream = new Random(SWAY_SALT);

    private SideDraw() {
    }

    /**
     * Re-bases the stream for a new match.
     *
     * @param seed The match seed the other generators were just given.
     */
    static void reseed(long seed) {
        stream = new Random(seed ^ SWAY_SALT);
    }

    /**
     * A uniform float in {@code [lo, hi)}, the engine's own {@code f.d}
     * contract, served off the sim stream.
     *
     * @param lo Inclusive lower bound.
     * @param hi Exclusive upper bound.
     * @return The draw.
     */
    public static float d(float lo, float hi) {
        return stream.nextFloat() * (hi - lo) + lo;
    }

    /**
     * The same contract under the engine's other name for it: {@code f.c(FF)F}
     * and {@code f.d(FF)F} have byte-identical bodies in the pinned jar, and
     * the retarget reuses the original call's NameAndType, so a rewired
     * {@code f.c} site can only land on a method NAMED {@code c}.
     *
     * @param lo Inclusive lower bound.
     * @param hi Exclusive upper bound.
     * @return The draw.
     */
    public static float c(float lo, float hi) {
        return stream.nextFloat() * (hi - lo) + lo;
    }
}
