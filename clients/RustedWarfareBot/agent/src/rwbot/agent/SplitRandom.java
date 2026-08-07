package rwbot.agent;

import java.util.Optional;
import java.util.Random;

/**
 * Splits the engine's one random stream in two: the simulation's, and
 * everyone else's.
 *
 * <p>The draw tap ran the last divergence to ground in stages. First it
 * proved every draw -- the opponent AI, unit simulation, particle spawns,
 * cosmetic sway -- happens on ONE thread, so a thread split routes nothing.
 * Then it named the wall-paced drawers: the ambient spawner
 * ({@code d.c.a}, silenced separately) and the unit sway redraws
 * ({@code units/e/b.a:260,265}), whose per-window counts wobbled 9-vs-6
 * between identical pinned runs. Their common shape: RENDER-path code fed
 * the measured wall-clock delta, drawing from the same generator the AI's
 * decisions use, at moments only the scheduler chooses
 * (wiki: policy-determinism).
 *
 * <p>The invariant that separates them is not the thread but the CALL PATH:
 * everything the simulation does runs under the engine's own tick --
 * {@code game.i.a(float)}, the method that increments the frame counter
 * ([[engine-tick-and-clock]]) -- and everything cosmetic reaches the
 * generator from the render loop ({@code java.u}) without passing through
 * it. So every draw walks its stack: through the tick means the seeded
 * stream this object inherits, otherwise the side stream. The simulation's
 * sequence becomes a pure function of the seed whatever the wall clock
 * does, and the cosmetics lose nothing, since their draws were never
 * reproducible to begin with.
 *
 * <p>The walk costs microseconds and the engine draws about once per tick,
 * so the price is noise. {@code setSeed} re-pins both streams, so the
 * match-start reseed keeps its meaning without knowing the split exists.
 */
class SplitRandom extends Random {

    private static final long serialVersionUID = 1L;

    /** Offsets the side stream's seed so the two streams never correlate. */
    private static final long SIDE_SALT = 0x51DE57E4A5EEDL;

    /** The tick owner: the class whose update increments the frame counter. */
    private static final String TICK_CLASS = "com.corrodinggames.rts.game.i";

    /**
     * The tick entries on {@link #TICK_CLASS}, by name and descriptor: the
     * engine's own pass calls {@code a(float, int)}, which locks and runs
     * {@code b(float, int)}, which runs {@code a(float)} -- the
     * {@code updateAllGame1} body that increments the frame counter
     * ([[engine-tick-and-clock]]). Descriptors matter: the same class also
     * carries {@code a(m.l, float)}, the world DRAW pass, and a name-only
     * match would hand every cosmetic draw the simulation's stream back.
     */
    private static final String[] TICK_NAMES = {"a", "b", "a"};

    /** Descriptors matching {@link #TICK_NAMES} position by position. */
    private static final String[] TICK_DESCRIPTORS = {"(FI)V", "(FI)V", "(F)V"};

    /** The render loop; reaching it before the tick means a cosmetic draw. */
    private static final String RENDER_CLASS = "com.corrodinggames.rts.java.u";

    /** One walker for every draw; descriptors need the class-reference option. */
    private static final StackWalker WALKER =
            StackWalker.getInstance(StackWalker.Option.RETAIN_CLASS_REFERENCE);

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
        if (underTick()) {
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

    /**
     * Whether the current draw is being made by the simulation.
     *
     * <p>Walks the caller's stack top-down: hitting the tick first means the
     * simulation (however it was invoked -- the engine's own pass or the
     * fast-forward's extra ticks); hitting the render loop first means a
     * cosmetic draw made outside any tick. A stack containing neither --
     * loading screens, menu threads -- is not the simulation.
     */
    private boolean underTick() {
        Optional<Boolean> verdict =
                WALKER.walk(
                        frames ->
                                frames.map(SplitRandom::classify)
                                        .filter(java.util.Objects::nonNull)
                                        .findFirst());
        return verdict.orElse(Boolean.FALSE).booleanValue();
    }

    /** One frame's vote: tick, render loop, or no opinion. */
    private static Boolean classify(StackWalker.StackFrame frame) {
        String owner = frame.getClassName();
        if (TICK_CLASS.equals(owner)) {
            for (int i = 0; i < TICK_NAMES.length; i++) {
                if (TICK_NAMES[i].equals(frame.getMethodName())
                        && TICK_DESCRIPTORS[i].equals(frame.getDescriptor())) {
                    return Boolean.TRUE;
                }
            }
            return null;
        }
        if (RENDER_CLASS.equals(owner)) {
            return Boolean.FALSE;
        }
        return null;
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
