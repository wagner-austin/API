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

    /**
     * Accumulators the AI's own init leaves alone; constructed default is zero.
     *
     * <p><b>Enumerated against the bytecode rather than collected by
     * observation, because the first version of this list was one field
     * short.</b> The AI spends the frame delta two ways and a grep for one
     * form finds only half of them:
     *
     * <pre>
     *   this.aX += f2;                  aL aM aN aO aP aQ aS
     *   this.aX = f.a(this.aX, f2);     aR aT aU aW
     * </pre>
     *
     * <p>{@code av()} restores {@code aL aN aP aT aU aW}; this list covers
     * the rest. {@code aR} belonged to neither and was decremented every
     * update ({@code a.java:1265}) after being set to 900 during play
     * ({@code a.java:1373}) -- including during the free ticks before
     * liveness, which carry the MEASURED delta. So it entered every match
     * with a wall-valued offset, and everything its cadence gates went with
     * it (wiki log 2026-08-31).
     */
    private static final String[] PLAIN_CLOCKS = {"aM", "aO", "aQ", "aR", "aS", "bG"};

    /**
     * The same clocks on the AI's SUB-controllers, which {@code av()} never
     * reaches and this class did not either.
     *
     * <p><b>Resetting the AI object was not enough, because the AI is not the
     * only thing that thinks.</b> {@code a.java:1383} drives every {@code h}
     * in the AI's {@code bm} list once per update, handing each the frame
     * delta, and both sub-controller classes spend it the same way the AI
     * does: {@code this.X = f.a(this.X, delta)} against a boundary, and when
     * one drains it acts and resets. One of those actions is a draw --
     * {@code i.java:889}, {@code f.a(0,100) < 5} -- which the draw tap named
     * as {@code game.a.i.b:2342}, the earliest differing call site across
     * four invocations with everything else already pinned (wiki log
     * 2026-08-30).
     *
     * <p>They start every run with a wall-valued offset for exactly the
     * reason the class doc gives for the AI's own: the free ticks before
     * liveness carry the MEASURED delta, because the delta pin lands at
     * liveness. Restored here to their CONSTRUCTED defaults, which are not
     * all zero -- {@code i.i} starts at 50, {@code i.g} at 100,
     * {@code n.f} at 4000 -- so a blanket zeroing would put the AI into a
     * state its constructor never produces.
     *
     * <p>Keyed by class because the two disagree about both which fields are
     * clocks and what they start at.
     */
    private static java.util.Map<String, java.util.Map<String, Float>> subClocks() {
        java.util.Map<String, java.util.Map<String, Float>> byClass =
                new java.util.LinkedHashMap<String, java.util.Map<String, Float>>();
        java.util.Map<String, Float> group = new java.util.LinkedHashMap<String, Float>();
        group.put("e", Float.valueOf(0.0f));
        group.put("g", Float.valueOf(100.0f));
        group.put("i", Float.valueOf(50.0f));
        group.put("j", Float.valueOf(50.0f));
        group.put("k", Float.valueOf(0.0f));
        group.put("m", Float.valueOf(0.0f));
        byClass.put(EngineNames.AI_GROUP_CLASS, group);

        java.util.Map<String, Float> task = new java.util.LinkedHashMap<String, Float>();
        task.put("f", Float.valueOf(4000.0f));
        task.put("g", Float.valueOf(100.0f));
        task.put("i", Float.valueOf(0.0f));
        task.put("j", Float.valueOf(0.0f));
        task.put("k", Float.valueOf(0.0f));
        byClass.put(EngineNames.AI_TASK_CLASS, task);
        return byClass;
    }

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
        int controllers = 0;
        for (int index = 0; index < count; index++) {
            Object team = EngineAccess.invoke(lookup, null, Integer.valueOf(index));
            if (team == null || !ai.isInstance(team)) {
                continue;
            }
            EngineAccess.invoke(init, team);
            for (String clock : PLAIN_CLOCKS) {
                EngineAccess.writeFloatField(team, clock, 0.0f);
            }
            controllers += resetSubControllers(team);
            restored++;
        }
        Log.info(
                "AI think-timers restored to constructed state for " + restored + " team(s) and "
                        + controllers + " sub-controller(s)");
    }

    /**
     * Restores one AI's sub-controllers' cadence clocks.
     *
     * <p>Matched on exact class name rather than on assignability: the two
     * tables disagree about which fields are clocks and what they start at,
     * so a subclass answering to a parent's table would be restored to a
     * state its own constructor never produces. A sub-controller of neither
     * class is left alone and counted as such by not being counted.
     *
     * @param team The AI whose sub-controllers are reset.
     * @return How many were restored.
     * @throws IllegalStateException When the pinned list field or any clock
     *     field is absent, which means the obfuscated layout moved.
     */
    private static int resetSubControllers(Object team) {
        Object held = EngineAccess.readField(team, EngineNames.AI_SUBCONTROLLERS);
        if (!(held instanceof Iterable)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.AI_CLASS + "." + EngineNames.AI_SUBCONTROLLERS
                            + " is not iterable" + EngineNames.PIN);
        }
        java.util.Map<String, java.util.Map<String, Float>> tables = subClocks();
        int reset = 0;
        // Named, not counted. A reset that restores nothing is
        // indistinguishable from an empty list unless the list says what was
        // in it -- and the first run of this reported "0 sub-controller(s)"
        // with no way to tell whether the AI had none yet or had some this
        // table does not name.
        java.util.List<String> seen = new java.util.ArrayList<String>();
        for (Object controller : (Iterable<?>) held) {
            if (controller == null) {
                continue;
            }
            seen.add(controller.getClass().getName());
            java.util.Map<String, Float> clocks = tables.get(controller.getClass().getName());
            if (clocks == null) {
                continue;
            }
            for (java.util.Map.Entry<String, Float> clock : clocks.entrySet()) {
                EngineAccess.writeFloatField(
                        controller, clock.getKey(), clock.getValue().floatValue());
            }
            reset++;
        }
        Log.info(
                "AI sub-controllers at liveness: " + seen.size() + " " + seen + "; "
                        + reset + " had cadence clocks restored");
        return reset;
    }
}
