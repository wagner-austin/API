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
        EngineRandom.swapEngineGenerator(installed, "the tick-split generator");
        Log.info(
                "tick-split generator installed: draws under the simulation tick are pinned;"
                        + " render-path draws use a side stream");
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

    /**
     * Called for every draw routed to the simulation stream; does nothing.
     *
     * <p>The seam the draw tap counts through: with routing live, the only
     * draws worth diffing between runs are the ones that can still desync
     * the sim (see RandomTap).
     */
    void onSimDraw() {
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
