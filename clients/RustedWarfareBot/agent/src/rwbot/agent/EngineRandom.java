package rwbot.agent;

import java.lang.reflect.Method;
import java.util.Random;

/**
 * Pins the engine's non-deterministic randomness so a run can be repeated.
 *
 * <p>Two runs of the same bot against the same map produced the best and the
 * worst results it had ever recorded, from identical code. That made every
 * measurement arguable: an A/B of a policy change cannot be read when the
 * between-run spread is larger than the effect (wiki: policy-combat).
 *
 * <p><b>The engine has two kinds of randomness and only one of them is the
 * problem.</b> Anything that must agree between peers goes through a
 * deterministic hash of the match seed and the frame counter -- no generator at
 * all, which is what lockstep requires and why the engine's own error string
 * for it reads {@code notRandInt}. Everything else goes through a plain
 * {@link Random} constructed with no seed, so it is seeded from the system
 * clock and differs every run.
 *
 * <p>The opponents' decisions are on the second path: the weighted unit mix
 * that chooses what they build, and the roll that decides whether an attack
 * group targets a position or a unit, both draw from it
 * (wiki: ai-opponent-strategy). Seeding it therefore makes their play
 * repeatable.
 *
 * <p><b>This does not make a run deterministic, and must not be described as
 * if it did.</b> The planner connects over a socket and reads samples on its
 * own schedule, so which frame an order lands on still varies. What seeding
 * removes is the largest and most obviously uncontrolled source, which is
 * enough to make repeated runs comparable rather than exactly equal.
 */
final class EngineRandom {

    private EngineRandom() {
    }

    /**
     * Seeds the engine's unseeded generator.
     *
     * <p>The field is {@code static final}, which is not an obstacle: nothing
     * is reassigned. The generator object is fetched and told to reseed itself
     * through {@link Random#setSeed(long)}, which is the same class the engine
     * would have called had it wanted a repeatable game.
     *
     * @param seed The seed to install.
     * @throws IllegalStateException When the pinned name is absent, or the
     *     field does not hold a {@link Random} -- either of which means the
     *     obfuscated layout moved and the seed would otherwise be silently
     *     ignored.
     */
    static void seed(long seed) {
        Class<?> holder = EngineAccess.pinnedClass(EngineNames.RANDOM_HOLDER_CLASS);
        Object generator;
        try {
            generator = EngineAccess.pinnedField(holder, EngineNames.RANDOM_FIELD).get(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read " + EngineNames.RANDOM_HOLDER_CLASS + "."
                            + EngineNames.RANDOM_FIELD + EngineNames.PIN, e);
        }
        if (!(generator instanceof Random)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.RANDOM_HOLDER_CLASS + "."
                            + EngineNames.RANDOM_FIELD + " is not a java.util.Random"
                            + EngineNames.PIN);
        }
        Method setSeed = EngineAccess.pinnedMethod(Random.class, "setSeed", long.class);
        EngineAccess.invoke(setSeed, generator, Long.valueOf(seed));
        Log.info("engine random seeded with " + seed + "; opponent choices are now repeatable");
    }
}
