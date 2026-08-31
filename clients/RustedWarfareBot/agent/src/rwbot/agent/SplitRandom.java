package rwbot.agent;

import java.util.Random;

/**
 * Splits the engine's one random stream in two: the simulation's, and
 * everyone else's.
 *
 * <p>The draw tap ran the wall-paced divergence to ground in stages. First
 * it proved every draw -- the opponent AI, unit simulation, particle
 * spawns, cosmetic sway -- happens on ONE thread, so a thread split routes
 * nothing. Then it named the wall-paced drawers: RENDER-path code fed the
 * measured wall-clock delta, drawing from the same generator the AI's
 * decisions use, at moments only the scheduler chooses
 * (wiki: policy-determinism).
 *
 * <p>The invariant that separates them is the PHASE: everything the
 * simulation does runs under the engine's own tick, and everything
 * cosmetic runs outside it. The first build classified each draw by
 * walking its stack for the tick entries -- and that walk was itself the
 * next divergence: a per-draw classification that consults the JIT-shaped
 * frame stream is process-varying in a way no seed can reach, and
 * same-seed runs from separate invocations forked at the first
 * consequential roll while parallel replicas agreed for whole panels
 * (wiki: policy-determinism, the cross-invocation arc). Routing now asks
 * {@link TickBracket#simPhase()} -- a flag raised and lowered by queue
 * rides whose ordering the engine itself guarantees -- so the simulation's
 * sequence is a pure function of the seed whatever the JIT, the scheduler
 * or the wall clock do. The cosmetics lose nothing: their draws were never
 * reproducible to begin with.
 *
 * <p>{@code setSeed} re-pins both streams, so the match-start reseed keeps
 * its meaning without knowing the split exists.
 */
class SplitRandom extends Random {

    private static final long serialVersionUID = 1L;

    /** Offsets the side stream's seed so the two streams never correlate. */
    private static final long SIDE_SALT = 0x51DE57E4A5EEDL;


    /** The stream every non-simulation draw is served from. */
    private final Side side;

    SplitRandom(long seed) {
        super(seed);
        this.side = new Side(seed ^ SIDE_SALT);
    }

    /**
     * Installs the split on the engine's generator holder.
     *
     * <p>Runs at match start, before the reseed -- which then reads the field
     * afresh and pins whatever it finds.
     *
     * @param seed The seed the simulation stream starts from.
     * @throws IllegalStateException When the swap cannot be made or did not
     *     stick.
     */
    static void install(long seed) {
        SplitRandom installed = new SplitRandom(seed);
        EngineRandom.swapGenerator(EngineRandom.Slot.ENGINE, installed, "the tick-split generator");
        Log.info(
                "tick-split generator installed: draws under the simulation tick are pinned;"
                        + " render-path draws use a side stream");
    }

    /**
     * Installs the same split on {@link Math#random()}'s global generator.
     *
     * <p><b>Seeding that generator was never enough, and the ledger proves
     * it.</b> Twelve engine call sites read {@link Math#random()}, and the
     * holder is JVM-global: the render path and the simulation drew from one
     * stream, with the render path drawing at moments only the scheduler
     * chooses. Replayed across eleven seeds on HPC3, every seed whose
     * {@code math=} ledger state agreed at frame 0 replicated bit-exact over
     * 250 samples and every seed whose state differed at frame 0 forked --
     * while the engine stream, split here since 2026-08-07, diverged only
     * AFTER the world had already forked in all three cases. Cause and
     * consequence, in that order (wiki log 2026-08-30).
     *
     * <p>Seeded plainly, and NOT salted apart from the engine split. The
     * reseed that follows ({@link EngineRandom#seed(long)}) pins every
     * generator to the match seed and would undo a construction-time salt
     * anyway, leaving the two halves of one object disagreeing about which
     * seed they were built from. All three generators have always taken the
     * same seed; what changes here is the phase routing, not the pinning.
     *
     * @param seed The seed the simulation stream starts from.
     * @throws IllegalStateException When the swap cannot be made or did not
     *     stick.
     */
    static void installMath(long seed) {
        SplitRandom installed = new SplitRandom(seed);
        EngineRandom.swapGenerator(
                EngineRandom.Slot.MATH, installed, "the Math.random tick-split generator");
        Log.info(
                "Math.random tick-split generator installed: the simulation's draws from it"
                        + " no longer share a stream with the render path");
    }

    /**
     * Installs the same split on {@link java.util.Collections#shuffle}'s
     * generator, the third and last of the shared streams.
     *
     * <p><b>It was invisible until the leak above it closed.</b> With Math
     * still shared, the shuffle stream agreed across all eleven replayed
     * seeds -- the world forked on Math first, so nothing downstream got to
     * matter. Split Math, and shuffle became the first stream to diverge, at
     * frame 600, ahead of the engine stream at 1050 and the world fork at
     * 3300 (wiki log 2026-08-30). Three generators, one seam, and the fix
     * only counts when every one of them routes by phase.
     *
     * @param seed The seed the simulation stream starts from.
     * @throws IllegalStateException When the swap cannot be made or did not
     *     stick.
     */
    static void installShuffle(long seed) {
        SplitRandom installed = new SplitRandom(seed);
        EngineRandom.swapGenerator(
                EngineRandom.Slot.SHUFFLE, installed, "the shuffle tick-split generator");
        Log.info(
                "Collections.shuffle tick-split generator installed: unit-mix order no longer"
                        + " shares a stream with anything outside the tick");
    }

    @Override
    protected int next(int bits) {
        // side is null exactly while the superclass constructor runs, and
        // Random(long) makes no draws; guarded so a future JDK changing that
        // cannot turn the split into a boot crash.
        if (side == null) {
            return super.next(bits);
        }
        if (TickBracket.simPhase()) {
            onSimDraw();
            return super.next(bits);
        }
        return side.draw(bits);
    }

    /** How many draws this generator has served to the simulation. */
    private long simDraws;

    /**
     * Called for every draw routed to the simulation stream; counts it.
     *
     * <p>The seam the draw tap counts through: with routing live, the only
     * draws worth diffing between runs are the ones that can still desync
     * the sim (see RandomTap).
     *
     * <p><b>Counting is what turns a stream state into a diagnosis.</b> Two
     * runs whose stream states differ have consumed a different number of
     * draws, but the state alone cannot say how many, so it cannot say
     * whether one extra draw slipped in or the whole sequence shifted. The
     * count can, and it is the number the remaining leak has to be found
     * through: with all three generators split, seed 31337's engine stream
     * still parts at frame 300 while the worlds agree to sample 116.
     *
     * <p>Not synchronised, and does not need to be: every draw the
     * simulation makes happens on the ticking thread, which is the same
     * invariant {@link TickBracket#simPhase()} already relies on.
     */
    void onSimDraw() {
        simDraws++;
    }

    /**
     * How many draws the simulation has taken from this generator.
     *
     * @return The running count.
     */
    long simDraws() {
        return simDraws;
    }

    @Override
    public synchronized void setSeed(long seed) {
        // Called by Random's own constructor before the fields exist, and by
        // the match-start reseed after; both streams follow the pin.
        super.setSeed(seed);
        if (side != null) {
            side.setSeed(seed ^ SIDE_SALT);
        }
    }

    /** The other stream, with the protected draw exposed to the split. */
    private static final class Side extends Random {

        private static final long serialVersionUID = 1L;

        Side(long seed) {
            super(seed);
        }

        int draw(int bits) {
            return next(bits);
        }
    }
}
