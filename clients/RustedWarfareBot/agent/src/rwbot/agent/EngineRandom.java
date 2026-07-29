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
 * <p><b>There are two unseeded generators, not one, and pinning only the
 * engine's left runs irreproducible.</b> Twelve call sites go through
 * {@link Math#random()} instead, which draws from a JVM-global generator that
 * seeding the engine's field cannot reach. They are not incidental: the AI
 * chooses <i>which unit to plant a new base at</i> through it
 * ({@code game/a/a.java:1713,1737,1761}), positions its sites and worker
 * destinations on a random disc around them ({@code game/a/o.java:96-97,166-167}
 * -- {@code o.w()}, whose result {@code a.java:1575} hands a worker as a
 * destination), and scatters unit positions by up to eight world units
 * ({@code game/units/y.java:4811-4837}).
 *
 * <p>So the opponents built their bases somewhere different every run. Two
 * matches from an identical job specification -- same seed, same arguments,
 * same code -- came back <i>survived, four extractors, 66 credits a second,
 * worth 13,800</i> and <i>defeated, no extractors, no income, worth 500</i>.
 * That spread is larger than every policy effect measured against it
 * (wiki: policy-determinism).
 *
 * <p>Both are pinned here. {@link Math}'s holder is lazily initialised, so it
 * is forced before being reached for, and reaching it needs
 * {@code --add-opens java.base/java.lang=ALL-UNNAMED} on the command line --
 * absent that, the reflective access throws rather than silently leaving the
 * generator unpinned.
 *
 * <p><b>This still does not make a run bit-for-bit deterministic, and must not
 * be described as if it did.</b> The planner connects over a socket and reads
 * samples on its own schedule, so which frame an order lands on still varies,
 * and the simulation advances by a wall-clock delta that CPU load perturbs.
 * What seeding removes is the dominant uncontrolled source.
 */
final class EngineRandom {

    private EngineRandom() {
    }

    /** Where {@link Math#random()} keeps its lazily built generator. */
    private static final String MATH_HOLDER = "java.lang.Math$RandomNumberGeneratorHolder";

    /** The field inside {@link #MATH_HOLDER} holding it. */
    private static final String MATH_FIELD = "randomNumberGenerator";

    /**
     * Seeds both unseeded generators: the engine's own and {@link Math}'s.
     *
     * @param seed The seed to install.
     * @throws IllegalStateException When either generator cannot be reached.
     */
    static void seed(long seed) {
        seedEngine(seed);
        seedMath(seed);
    }

    /**
     * Seeds the generator the engine's own helper draws from.
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
    private static void seedEngine(long seed) {
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

    /**
     * Seeds the JVM-global generator behind {@link Math#random()}.
     *
     * <p>Twelve engine call sites use it, including the AI's choice of where to
     * plant a base and where to send a worker, so leaving it unpinned is what
     * made two runs of one job specification disagree about the whole match.
     *
     * <p>The holder is initialised on first use, so {@link Math#random()} is
     * called once before the field is read -- reading it beforehand would find
     * the class uninitialised and construct nothing to seed.
     *
     * @param seed The seed to install.
     * @throws IllegalStateException When the holder cannot be reached, which on
     *     this JVM means {@code --add-opens java.base/java.lang=ALL-UNNAMED}
     *     was not passed. Failing loudly is deliberate: a silently unpinned
     *     generator reads as a reproducible run that is not one.
     */
    private static void seedMath(long seed) {
        Math.random();
        Object generator;
        try {
            java.lang.reflect.Field field =
                    Class.forName(MATH_HOLDER).getDeclaredField(MATH_FIELD);
            field.setAccessible(true);
            generator = field.get(null);
        } catch (ClassNotFoundException | NoSuchFieldException | IllegalAccessException
                | RuntimeException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot reach " + MATH_HOLDER + "." + MATH_FIELD
                            + "; pass --add-opens java.base/java.lang=ALL-UNNAMED", e);
        }
        if (!(generator instanceof Random)) {
            throw new IllegalStateException(
                    "rw-agent: " + MATH_HOLDER + "." + MATH_FIELD + " is not a java.util.Random");
        }
        ((Random) generator).setSeed(seed);
        Log.info("Math.random seeded with " + seed + "; opponent placement is now repeatable");
    }
}
