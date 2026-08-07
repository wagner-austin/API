package rwbot.agent;

import java.lang.reflect.Method;

/**
 * Runs the simulation faster than the wall clock, in identical steps.
 *
 * <p><b>The mechanism, paid for with eight instrumented nulls.</b> This
 * engine does not advance through Slick's fixed-timestep drain: the
 * simulation ticks ONCE PER RENDER PASS -- {@code u.render} calls
 * {@code l.a(delta * 0.06f, deltaMs)} and later drains the script queue our
 * hook rides -- so game speed is pass rate times the engine's own delta
 * clamp ({@code l.bu}, the determinism pin), and every container-side lever
 * (target frame rate, accumulator injection, smooth-delta spoofing) lands
 * on machinery the engine ignores (task #35, log 2026-08-06).
 *
 * <p>So the accelerator calls the tick itself: up to {@code multiplier - 1}
 * extra invocations of the engine's own tick entry per pass, each advancing
 * the same pinned quantum the clamp enforces, and NEVER past the next
 * lockstep sample boundary -- samples land on exactly the frames a realtime
 * run samples, which is what makes the world-digest comparison meaningful
 * ([[policy-determinism]]).
 *
 * <p>Reentrancy is safe by measurement, not hope: the script queue drains
 * from {@code u.render} directly (stack-proven), not from inside the tick,
 * so extra ticks issued here never re-enter the hook that issued them.
 */
final class FastForward {

    private static Method tick;
    private static int extraPerPass;
    private static float stepScaled;
    private static int stepMsHeld;
    private static long windowStartNs;
    private static int passesInWindow;
    private static long framesAtWindowStart;

    private FastForward() {
    }

    /**
     * Arms the accelerator.
     *
     * @param engine The live engine instance whose tick will be driven.
     * @param multiplier Wall-clock multiple to run at; 1 or less arms
     *     nothing, because one tick per pass is what the engine already
     *     does.
     * @param stepMs The pinned logic step, passed to each extra tick so the
     *     engine clock advances exactly as a realtime run's does.
     */
    static void arm(Object engine, int multiplier, int stepMs) {
        if (multiplier <= 1) {
            return;
        }
        tick = EngineAccess.pinnedMethod(engine.getClass(), "a", float.class, int.class);
        extraPerPass = multiplier - 1;
        stepScaled = stepMs * 0.06f;
        stepMsHeld = stepMs;
        Log.info(
                "fast-forward armed: " + multiplier + "x -- up to " + extraPerPass
                        + " extra engine tick(s) per render pass, boundary-aligned");
    }

    /**
     * Advances the simulation toward the next sample boundary.
     *
     * <p>Called on the game thread once per pass, before the boundary test.
     * Never ticks past the boundary: the sample must land on the same frame
     * a realtime run samples, or the two runs' traces stop being the same
     * experiment.
     *
     * @param engine The live engine instance.
     * @param frame The frame counter as of this pass.
     * @param boundary The next sample frame, or any value at or below
     *     {@code frame} to decline.
     * @return The frame counter after any extra ticks.
     */
    static int advanceToward(Object engine, int frame, int boundary) {
        passesInWindow++;
        if (tick == null || boundary <= frame) {
            report(frame);
            return frame;
        }
        int budget = Math.min(extraPerPass, boundary - frame - 1);
        for (int i = 0; i < budget; i++) {
            EngineAccess.invoke(
                    tick, engine, Float.valueOf(stepScaled), Integer.valueOf(stepMsHeld));
        }
        int advanced =
                budget <= 0 ? frame : EngineAccess.readIntField(engine, StateStream.FRAME_FIELD);
        report(advanced);
        return advanced;
    }

    /** Once a second: passes and effective frames, the two speeds that matter. */
    private static void report(int frame) {
        long now = System.nanoTime();
        if (windowStartNs == 0L) {
            windowStartNs = now;
            framesAtWindowStart = frame;
        } else if (now - windowStartNs >= 1_000_000_000L) {
            if (tick != null) {
                Log.info(
                        "fast-forward diag: "
                                + passesInWindow
                                + " passes/s, "
                                + (frame - framesAtWindowStart)
                                + " frames/s");
            }
            passesInWindow = 0;
            framesAtWindowStart = frame;
            windowStartNs = now;
        }
    }
}
