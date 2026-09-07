package rwbot.agent;

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
 * clock and differs every run. The engine even ships a per-match reset --
 * {@code f.a()} seeds its second generator to zero -- but every draw helper
 * reads the FIRST generator, which that reset never touches (bytecode-verified
 * against the pinned jar): the engine seeds a generator it does not use and
 * uses a generator it never seeds.
 *
 * <p>The opponents' decisions are on the unseeded path: the weighted unit mix
 * that chooses what they build, and the roll that decides whether an attack
 * group targets a position or a unit, both draw from it
 * (wiki: ai-opponent-strategy). Seeding it therefore makes their play
 * repeatable.
 *
 * <p><b>There are three unseeded generators, not one, and pinning fewer left
 * runs irreproducible.</b> Twelve call sites go through {@link Math#random()},
 * which draws from a JVM-global generator that seeding the engine's field
 * cannot reach. They are not incidental: the AI chooses <i>which unit to plant
 * a new base at</i> through it ({@code game/a/a.java:1713,1737,1761}),
 * positions its sites and worker destinations on a random disc around them
 * ({@code game/a/o.java:96-97,166-167} -- {@code o.w()}, whose result
 * {@code a.java:1575} hands a worker as a destination), and scatters unit
 * positions by up to eight world units ({@code game/units/y.java:4811-4837}).
 * And the AI's unit-mix rebuild shuffles its candidate list through
 * {@link java.util.Collections#shuffle(java.util.List)}
 * ({@code game/a/d.java:43}), which draws from Collections' own lazily built
 * generator -- a third stream the other two pins never reach.
 *
 * <p>So the opponents built their bases somewhere different every run. Two
 * matches from an identical job specification -- same seed, same arguments,
 * same code -- came back <i>survived, four extractors, 66 credits a second,
 * worth 13,800</i> and <i>defeated, no extractors, no income, worth 500</i>.
 * That spread is larger than every policy effect measured against it
 * (wiki: policy-determinism).
 *
 * <p>All three are pinned here. {@link Math}'s holder is lazily initialised, so
 * it is forced before being reached for, and reaching it needs
 * {@code --add-opens java.base/java.lang=ALL-UNNAMED} on the command line;
 * Collections' generator likewise, behind
 * {@code --add-opens java.base/java.util=ALL-UNNAMED}. Absent either flag the
 * reflective access throws rather than silently leaving a generator unpinned.
 *
 * <p><b>This still does not make a run bit-for-bit deterministic, and must not
 * be described as if it did.</b> The planner connects over a socket and reads
 * samples on its own schedule, so which frame an order lands on still varies,
 * and the simulation advances by a wall-clock delta that CPU load perturbs.
 * What seeding removes is the dominant uncontrolled source. Whether the pinned
 * streams are then consumed identically run to run is exactly what
 * {@link RandomLedger} exists to measure.
 */
final class EngineRandom {

    private EngineRandom() {
    }

    /** Where {@link Math#random()} keeps its lazily built generator. */
    private static final String MATH_HOLDER = "java.lang.Math$RandomNumberGeneratorHolder";

    /** The field inside {@link #MATH_HOLDER} holding it. */
    private static final String MATH_FIELD = "randomNumberGenerator";

    /** Where {@link java.util.Collections#shuffle(java.util.List)} keeps its generator. */
    private static final String SHUFFLE_FIELD = "r";

    /**
     * Seeds all three unseeded generators: the engine's own, {@link Math}'s,
     * and {@link java.util.Collections}'.
     *
     * @param seed The seed to install.
     * @throws IllegalStateException When any generator cannot be reached.
     */
    static void seed(long seed) {
        engineGenerator().setSeed(seed);
        Log.info("engine random seeded with " + seed + "; opponent choices are now repeatable");
        mathGenerator().setSeed(seed);
        Log.info("Math.random seeded with " + seed + "; opponent placement is now repeatable");
        shuffleGenerator().setSeed(seed);
        Log.info("Collections.shuffle seeded with " + seed + "; unit-mix order is now repeatable");
        SideDraw.reseed(seed);
        Log.info("sway side stream re-based; rewired draw sites no longer touch the sim stream");
    }

    /**
     * The generator the engine's own draw helpers read.
     *
     * <p>The field is {@code static final}, which is not an obstacle: nothing
     * is reassigned. The generator object is fetched and reseeded through
     * {@link Random#setSeed(long)}, which is the same call the engine's own
     * per-match reset makes -- against the wrong field (see class doc).
     *
     * @return The live generator instance.
     * @throws IllegalStateException When the pinned name is absent, or the
     *     field does not hold a {@link Random} -- either of which means the
     *     obfuscated layout moved and the seed would otherwise be silently
     *     ignored.
     */
    static Random engineGenerator() {
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
        return (Random) generator;
    }

    /**
     * The JVM-global generator behind {@link Math#random()}.
     *
     * <p><b>Reached without drawing, and that wording is a correction paid
     * for in contaminated measurements.</b> This method used to call
     * {@link Math#random()} first "so the holder is initialised before the
     * field is read" -- one draw from the very stream every caller was
     * trying to OBSERVE, on every call. The ledger's per-sample state line
     * paid it invisibly for weeks (deterministic cadence, so twins agreed),
     * and the per-tick pinpoint probe multiplied it 75-fold and measured
     * its own draws as the engine's (wiki log, the 2026-09-07 correction
     * entry; the extended tap named {@code RandomLedger.describe:47} from
     * the live stack). The {@code Class.forName} below already initialises
     * the holder -- {@code initialize} defaults to true and the holder's
     * static init constructs the generator without drawing -- so the draw
     * bought nothing the reflection did not already have.
     *
     * @return The live generator instance.
     * @throws IllegalStateException When the holder cannot be reached, which on
     *     this JVM means {@code --add-opens java.base/java.lang=ALL-UNNAMED}
     *     was not passed. Failing loudly is deliberate: a silently unpinned
     *     generator reads as a reproducible run that is not one.
     */
    static Random mathGenerator() {
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
        return (Random) generator;
    }

    /**
     * The three generator slots the simulation draws from.
     *
     * <p><b>One shape, three holders.</b> Each is a {@code static final}
     * field holding a {@link Random}, each is lazily built, and each has to
     * be forced into existence before it can be swapped and read back after.
     * Written out three times, the differences that matter -- which class,
     * which field, what forces it -- were buried in three copies of the
     * identical Unsafe-write-and-verify dance.
     *
     * <p>{@link #read()} does double duty deliberately: called before a
     * write it forces the lazy initialisation, so the holder cannot run its
     * own {@code <clinit>} over the replacement afterwards, and called after
     * it is the read-back. A premain swap was once silently overwritten
     * exactly that way (see RandomTap), and a replacement that did not stick
     * must fail the run rather than report a clean one.
     */
    enum Slot {

        /** The engine's own generator, read by its five draw helpers. */
        ENGINE {
            @Override
            Class<?> holder() {
                return EngineAccess.pinnedClass(EngineNames.RANDOM_HOLDER_CLASS);
            }

            @Override
            String field() {
                return EngineNames.RANDOM_FIELD;
            }

            @Override
            Random read() {
                return engineGenerator();
            }
        },

        /** {@link Math#random()}'s JVM-global generator, twelve call sites. */
        MATH {
            @Override
            Class<?> holder() {
                try {
                    return Class.forName(MATH_HOLDER);
                } catch (ClassNotFoundException e) {
                    throw new IllegalStateException(
                            "rw-agent: cannot reach " + MATH_HOLDER
                                    + "; pass --add-opens java.base/java.lang=ALL-UNNAMED", e);
                }
            }

            @Override
            String field() {
                return MATH_FIELD;
            }

            @Override
            Random read() {
                return mathGenerator();
            }
        },

        /** {@link java.util.Collections#shuffle}'s generator, unit-mix order. */
        SHUFFLE {
            @Override
            Class<?> holder() {
                return java.util.Collections.class;
            }

            @Override
            String field() {
                return SHUFFLE_FIELD;
            }

            @Override
            Random read() {
                return shuffleGenerator();
            }
        };

        /** The class declaring this slot's field. */
        abstract Class<?> holder();

        /** The field name inside {@link #holder()}. */
        abstract String field();

        /** Forces the slot's lazy initialisation and returns what it holds. */
        abstract Random read();
    }

    /**
     * Replaces one generator and verifies the swap stuck.
     *
     * <p>The field is {@code static final}, which reflection refuses to
     * write, so the write goes through {@code sun.misc.Unsafe} -- reached
     * reflectively rather than imported, because javac treats the name
     * itself as a warning and the build treats warnings as errors.
     *
     * @param slot Which generator to replace.
     * @param replacement The generator to install.
     * @param what What is being installed, for the failure message.
     * @throws IllegalStateException When Unsafe or the field cannot be
     *     reached, or the read-back does not return the replacement.
     */
    static void swapGenerator(Slot slot, Random replacement, String what) {
        // Before the write, so the holder cannot initialise over it after.
        slot.read();
        Class<?> holder = slot.holder();
        putStaticObject(holder, slot.field(), replacement, what);
        if (slot.read() != replacement) {
            throw new IllegalStateException(
                    "rw-agent: " + what + " did not stick on " + holder.getName() + "."
                            + slot.field() + "; the holder re-initialised over it");
        }
    }

    /**
     * Writes one {@code static final} object field.
     *
     * <p>Reflection on a static final field refuses the write, so it goes
     * through {@code sun.misc.Unsafe} -- reached reflectively rather than
     * imported, because javac treats the name itself as a warning and the
     * build treats warnings as errors.
     *
     * <p>Shared by both generator swaps rather than written twice: they
     * differ in which holder they target and in how they read back, and not
     * in how the write is made.
     *
     * @param holder The class declaring the field.
     * @param fieldName The field.
     * @param value What to write.
     * @param what What is being installed, for the failure message.
     * @throws IllegalStateException When Unsafe or the field cannot be
     *     reached.
     */
    private static void putStaticObject(
            Class<?> holder, String fieldName, Object value, String what) {
        try {
            Class<?> unsafeClass = Class.forName("sun.misc.Unsafe");
            java.lang.reflect.Field theUnsafe = unsafeClass.getDeclaredField("theUnsafe");
            theUnsafe.setAccessible(true);
            Object unsafe = theUnsafe.get(null);
            java.lang.reflect.Field field = holder.getDeclaredField(fieldName);
            Object base =
                    unsafeClass
                            .getMethod("staticFieldBase", java.lang.reflect.Field.class)
                            .invoke(unsafe, field);
            Object offset =
                    unsafeClass
                            .getMethod("staticFieldOffset", java.lang.reflect.Field.class)
                            .invoke(unsafe, field);
            unsafeClass
                    .getMethod("putObject", Object.class, long.class, Object.class)
                    .invoke(unsafe, base, offset, value);
        } catch (ClassNotFoundException | NoSuchFieldException | NoSuchMethodException
                | IllegalAccessException | java.lang.reflect.InvocationTargetException
                | RuntimeException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot install " + what + " on "
                            + holder.getName() + "." + fieldName, e);
        }
    }

    /**
     * The generator behind {@link java.util.Collections#shuffle(java.util.List)}.
     *
     * <p>Lazily built on the first single-argument shuffle, so one is forced on
     * an empty list -- a no-op reorder that exists purely to make the field
     * non-null -- before the field is read.
     *
     * @return The live generator instance.
     * @throws IllegalStateException When the field cannot be reached, which on
     *     this JVM means {@code --add-opens java.base/java.util=ALL-UNNAMED}
     *     was not passed.
     */
    static Random shuffleGenerator() {
        java.util.Collections.shuffle(new java.util.ArrayList<Object>());
        Object generator;
        try {
            java.lang.reflect.Field field =
                    java.util.Collections.class.getDeclaredField(SHUFFLE_FIELD);
            field.setAccessible(true);
            generator = field.get(null);
        } catch (NoSuchFieldException | IllegalAccessException | RuntimeException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot reach java.util.Collections." + SHUFFLE_FIELD
                            + "; pass --add-opens java.base/java.util=ALL-UNNAMED", e);
        }
        if (!(generator instanceof Random)) {
            throw new IllegalStateException(
                    "rw-agent: java.util.Collections." + SHUFFLE_FIELD
                            + " is not a java.util.Random");
        }
        return (Random) generator;
    }
}
