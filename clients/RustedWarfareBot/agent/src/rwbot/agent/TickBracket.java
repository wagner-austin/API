package rwbot.agent;

/**
 * Marks, from outside the engine, exactly when the simulation is ticking.
 *
 * <p>The split generator needs to know whether a draw belongs to the
 * simulation or to the render path, and it used to decide by walking the
 * caller's stack per draw. That walk was the last nondeterministic
 * classifier left in the sim path: same-seed runs from separate invocations
 * forked at the first consequential roll (frame 7050 on duel_lake, realtime
 * and 10x alike) while parallel replicas stayed bit-exact for 400 samples --
 * after every other candidate had been pinned or disproven by measurement
 * (wiki: policy-determinism, the cross-invocation arc). A classification
 * that consults the JIT-shaped world per draw is process-varying in a way
 * no seed can reach; this bracket replaces it with two queue rides whose
 * ordering the engine itself guarantees.
 *
 * <p>The cycle: an ENTER runnable rides the engine's pre-tick queue, which
 * drains at the top of the tick body before the world updates; it raises
 * the flag and posts its successor to the script queue, which drains after
 * the simulation in the same pass; the successor lowers the flag and posts
 * the next ENTER. Each hop crosses to the other queue, so neither runnable
 * is picked up by the drain it runs in -- the same-drain spin the
 * wrong-world guard paid to learn about. Draws between ENTER and EXIT are
 * the simulation's; everything else -- render, GUI, menu, loader -- is not.
 *
 * <p>Fast-forward's extra ticks run inside the script drain, after this
 * pass's EXIT may already have fired, so {@link FastForward} brackets its
 * own invocations explicitly; the queue-order ambiguity inside one drain
 * ends with the flag down either way, and the next tick's ENTER re-arms it.
 *
 * <p>Started once, at match liveness, right after the split installs --
 * pre-latch draws belong to the engine's original generator and are erased
 * by the reseed regardless.
 */
final class TickBracket {

    /** The one thread the simulation ticks on, captured at the first ENTER. */
    private static volatile Thread tickThread;

    /** Raised at the top of the tick body, lowered after the simulation. */
    private static volatile boolean inTick;

    private TickBracket() {
    }

    /**
     * Starts the self-sustaining enter/exit cycle on the live engine.
     *
     * @param engine The engine whose pre-tick queue the cycle rides.
     */
    static void start(Object engine) {
        Orders.onEngineTick(engine, TickBracket::enterAndChain);
        Log.info("tick bracket armed: sim draws route by phase, not by stack");
    }

    /**
     * Whether the current draw is the simulation's.
     *
     * <p>True exactly between ENTER and EXIT on the ticking thread. Any
     * other thread answers false whatever the flag says: a draw from a
     * loader or probe thread was never the simulation's, and the flag is
     * not theirs to read.
     */
    static boolean simPhase() {
        return inTick && Thread.currentThread() == tickThread;
    }

    /**
     * Raises the flag for work the caller is about to run on the ticking
     * thread itself -- fast-forward's extra tick invocations. Also claims
     * the thread: the first pass's extras can run before the first ENTER
     * has ever fired, and a bracket that only half-armed would route that
     * pass's sim draws to the side stream.
     */
    static void enterExtra() {
        tickThread = Thread.currentThread();
        inTick = true;
    }

    /** Lowers the flag after the caller's extra ticks. */
    static void exitExtra() {
        inTick = false;
    }

    /** The ENTER half: top of the tick body, before the world updates. */
    private static void enterAndChain() {
        tickThread = Thread.currentThread();
        inTick = true;
        Orders.onGameThread(TickBracket::exitAndChain);
    }

    /** The EXIT half: the script drain, after the simulation ran. */
    private static void exitAndChain() {
        inTick = false;
        Orders.onEngineTick(EngineHandle.current(), TickBracket::enterAndChain);
    }
}
