package rwbot.agent;

import java.lang.reflect.Method;

/**
 * Restores every opponent AI's think-timers to their constructed state at
 * match start.
 *
 * <p>The last run-to-run divergence the composed draw tap could still find:
 * between world creation and the hold latching, the world runs a stable
 * number of free ticks -- but those ticks carry the MEASURED delta, because
 * the delta pin lands at liveness. The AI's cadence accumulators
 * ({@code aP += f2} against a 25-unit boundary, and friends) therefore start
 * every run with a slightly different wall-valued offset. The world digest
 * cannot see a float accumulator, so runs stayed bit-identical for thousands
 * of frames -- until a think boundary landed one frame apart, one group roll
 * happened in different sample windows ({@code a.n.a:709}, 1-vs-0 at the
 * first window), and every subsequent draw cascaded
 * (wiki: policy-determinism).
 *
 * <p>So at the same moment the frame counter is zeroed, each AI's timers go
 * back to what its constructor made them: its own init ({@code av()}) is
 * invoked -- restoring the deliberately staggered cadences
 * ({@code aL = 100 + k*9} and so on) exactly as shipped -- and the
 * accumulators that init does not touch are zeroed to their field defaults.
 * Wall time stops being an input to when the AI thinks.
 */
final class AiTimers {

    private AiTimers() {
    }

    /** Accumulators the AI's own init leaves alone; constructed default is zero. */
    private static final String[] PLAIN_CLOCKS = {"aM", "aO", "aQ", "aS", "bG"};

    /**
     * Resets every AI team's think-timers to constructed state.
     *
     * <p>Runs on the game thread at match liveness, beside the frame zeroing
     * and the reseed.
     */
    static void reset() {
        Class<?> teams = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Class<?> ai = EngineAccess.pinnedClass(EngineNames.AI_CLASS);
        Method lookup = EngineAccess.pinnedMethod(teams, EngineNames.TEAM_LOOKUP, int.class);
        Method init = EngineAccess.pinnedMethod(ai, EngineNames.AI_INIT);
        int count = EngineAccess.readStaticIntField(teams, EngineNames.TEAM_COUNT);
        int restored = 0;
        for (int index = 0; index < count; index++) {
            Object team = EngineAccess.invoke(lookup, null, Integer.valueOf(index));
            if (team == null || !ai.isInstance(team)) {
                continue;
            }
            EngineAccess.invoke(init, team);
            for (String clock : PLAIN_CLOCKS) {
                EngineAccess.writeFloatField(team, clock, 0.0f);
            }
            restored++;
        }
        Log.info("AI think-timers restored to constructed state for " + restored + " team(s)");
    }
}
