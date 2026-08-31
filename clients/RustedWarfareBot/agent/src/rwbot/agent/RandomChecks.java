package rwbot.agent;

import java.util.Random;

/**
 * Checks that every generator the simulation draws from is phase-split.
 *
 * <p><b>Why a check and not a comment.</b> The engine's own generator was
 * split on 2026-08-07 and {@link Math#random()}'s was left shared, seeded but
 * unrouted -- and nothing failed, because seeding makes a run START from a
 * known state and says nothing about whether the simulation's draws from it
 * are a function of the seed. It cost a 24-member cluster panel to find:
 * across eleven seeds replayed on HPC3, every seed whose {@code math=}
 * ledger state agreed at frame 0 replicated bit-exact over 250 samples, and
 * every seed whose state differed at frame 0 forked, with the engine stream
 * diverging only after the world already had (wiki log 2026-08-30).
 *
 * <p>So the invariant is asserted here rather than remembered: a generator
 * the simulation reads must route by {@link TickBracket#simPhase()}, and the
 * two phases must not hand out the same numbers.
 */
final class RandomChecks {

    private RandomChecks() {
    }

    /** A seed with no special properties; any fixed value serves. */
    private static final long SEED = 31337L;

    /** How many draws each phase takes when the two are compared. */
    private static final int DRAWS = 16;

    /** Exercises the split's routing in both phases. */
    static int checkSplitRouting() {
        int failures = 0;

        failures += Check.expect(!TickBracket.simPhase(), "bracket down before any enter");

        // Same seed, drawn entirely in one phase then entirely in the other.
        // A split that ignored the phase would return one sequence twice.
        long[] sim = drawsInSimPhase(new SplitRandom(SEED));
        long[] side = drawsOutsideSimPhase(new SplitRandom(SEED));
        failures += Check.expect(!java.util.Arrays.equals(sim, side),
                "sim-phase and render-phase draws come from different streams");

        // And the sim stream is a pure function of the seed: a second
        // generator, having served a different number of render draws
        // first, still answers the simulation identically. This is the
        // property Math.random did not have -- its stream advanced with the
        // render path, so the simulation's draws depended on how many
        // frames had been drawn before them.
        SplitRandom polluted = new SplitRandom(SEED);
        drawsOutsideSimPhase(polluted);
        drawsOutsideSimPhase(polluted);
        long[] afterPollution = drawsInSimPhase(polluted);
        failures += Check.expect(java.util.Arrays.equals(sim, afterPollution),
                "render-path draws do not move the simulation's stream");

        // The reseed re-pins both halves, so a match-start reseed keeps its
        // meaning on a generator that has already served draws.
        SplitRandom reseeded = new SplitRandom(SEED ^ 0xFFFFL);
        drawsInSimPhase(reseeded);
        reseeded.setSeed(SEED);
        failures += Check.expect(java.util.Arrays.equals(sim, drawsInSimPhase(reseeded)),
                "setSeed re-pins the simulation stream");

        return failures;
    }

    /**
     * Draws with the bracket raised, as the simulation does.
     *
     * @param random The generator to draw from.
     * @return The values drawn, in order.
     */
    private static long[] drawsInSimPhase(Random random) {
        TickBracket.enterExtra();
        try {
            return draw(random);
        } finally {
            TickBracket.exitExtra();
        }
    }

    /**
     * Draws with the bracket lowered, as the render path does.
     *
     * @param random The generator to draw from.
     * @return The values drawn, in order.
     */
    private static long[] drawsOutsideSimPhase(Random random) {
        TickBracket.exitExtra();
        return draw(random);
    }

    /**
     * Takes a fixed number of draws.
     *
     * @param random The generator to draw from.
     * @return The values drawn, in order.
     */
    private static long[] draw(Random random) {
        long[] values = new long[DRAWS];
        for (int i = 0; i < DRAWS; i++) {
            values[i] = random.nextLong();
        }
        return values;
    }
}
