package rwbot.agent;

import java.lang.reflect.Field;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Reads the internal state of every pinned generator, so two runs can be
 * compared stream by stream.
 *
 * <p>Eight same-specification runs agreed bit-for-bit to frame 7200 and then
 * split into exactly two families on one opponent build choice, with every
 * generator seeded (wiki: policy-determinism). A world digest can only say
 * THAT the runs forked; it cannot say which random stream desynced, or
 * whether one did at all. This can: the state of a {@link Random} advances on
 * every draw, so two runs whose worlds still agree but whose generator states
 * differ have consumed a different number of draws -- the divergence is
 * located to a stream and a sample window long before it surfaces as a
 * different unit on the map. And if the worlds fork while every stream still
 * agrees, the cause is not a generator at all, which is just as decisive.
 *
 * <p>The state is {@link Random}'s private scrambled seed, read reflectively
 * -- an opaque identity, not a value anything interprets. Reaching it needs
 * {@code --add-opens java.base/java.util=ALL-UNNAMED}; absent that the read
 * throws rather than logging a ledger that silently compares nothing.
 */
final class RandomLedger {

    private RandomLedger() {
    }

    /** The scrambled-state field inside {@link Random}. */
    private static final String SEED_FIELD = "seed";

    /** Cached accessor for {@link #SEED_FIELD}; reflection once, reads forever. */
    private static Field seedField;

    /**
     * One line of generator states, in a fixed order for diffing.
     *
     * @return {@code engine=<hex> math=<hex> shuffle=<hex>}.
     * @throws IllegalStateException When any generator or its state cannot be
     *     reached.
     */
    static String describe() {
        Random engine = EngineRandom.engineGenerator();
        Random math = EngineRandom.mathGenerator();
        Random shuffle = EngineRandom.shuffleGenerator();
        return "engine=" + Long.toHexString(stateOf(engine))
                + " math=" + Long.toHexString(stateOf(math))
                + " shuffle=" + Long.toHexString(stateOf(shuffle))
                + " draws=" + simDrawsOf(engine)
                + "/" + simDrawsOf(math)
                + "/" + simDrawsOf(shuffle);
    }

    /**
     * How many draws the simulation has taken from one generator, or -1.
     *
     * <p><b>A state says two runs differ; a count says by how much.</b> The
     * state is a scrambled seed, so two runs that have consumed a different
     * number of draws are merely unequal -- nothing in the value says
     * whether one extra draw slipped in or the sequence shifted wholesale.
     * The count answers that directly, which is what the remaining leak
     * needs: with all three generators split, seed 31337's engine stream
     * still parts at frame 300 while the worlds agree to sample 116.
     *
     * @param generator The generator to read.
     * @return Its simulation-draw count, or -1 when it is not a split
     *     generator -- which is the honest answer for an unsplit stream
     *     rather than a zero that would read as "took no draws".
     */
    private static long simDrawsOf(Random generator) {
        if (generator instanceof SplitRandom) {
            return ((SplitRandom) generator).simDraws();
        }
        return -1;
    }

    /**
     * The scrambled internal state of one generator.
     *
     * @param generator The generator to read.
     * @return Its current state bits.
     * @throws IllegalStateException When the field cannot be reached, which on
     *     this JVM means {@code --add-opens java.base/java.util=ALL-UNNAMED}
     *     was not passed.
     */
    private static long stateOf(Random generator) {
        Object state;
        try {
            if (seedField == null) {
                Field field = Random.class.getDeclaredField(SEED_FIELD);
                field.setAccessible(true);
                seedField = field;
            }
            state = seedField.get(generator);
        } catch (NoSuchFieldException | IllegalAccessException | RuntimeException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read java.util.Random." + SEED_FIELD
                            + "; pass --add-opens java.base/java.util=ALL-UNNAMED", e);
        }
        if (!(state instanceof AtomicLong)) {
            throw new IllegalStateException(
                    "rw-agent: java.util.Random." + SEED_FIELD + " is not an AtomicLong");
        }
        return ((AtomicLong) state).get();
    }
}
