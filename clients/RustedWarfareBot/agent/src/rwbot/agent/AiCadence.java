package rwbot.agent;

/**
 * One line per sample of the AI's think-cadence state, for the draw tap's
 * diagnosis and for nothing else.
 *
 * <p>The tap ({@link RandomTap}) names the call site whose draw count varies
 * between byte-identical invocations; this names what FED that call site. The
 * 2026-09-06 tapped pairs forked on whether the AI's evaluation passes ran at
 * all in a window — one twin consulting its candidate gate at frame 75 and
 * the other first at frame 51,450 — with every named clock already restored
 * at liveness and the streams reseeded identically. What remains is the state
 * this line records per sample: the map's scanned pool count (the expansion
 * chooser draws a random index into that list and silently declines while it
 * is empty), each AI team's {@code aT}/{@code aU} creation clocks, and its
 * sub-controller roster size. Two runs' logs, lined up frame by frame, either
 * show these diverging BEFORE the first divergent draw — naming the seam —
 * or show them identical, which eliminates this layer the way the roster
 * snapshot and identity hashes were eliminated before it
 * (wiki: policy-determinism).
 *
 * <p>Same discipline as {@link AiZones} and the tap itself: agent log only,
 * never on the wire, armed by the tap's own option, inert otherwise. A
 * planner cannot consume it without deliberately adding a record kind.
 */
final class AiCadence {

    /**
     * Ticks logged so far by the early-window tracker; stops at
     * {@link #EARLY_WINDOW}. One counter, on the tick thread only.
     */
    private static int earlyTicks;

    /**
     * How many ticks the per-tick tracker records from liveness. The E/F and
     * G/H pairs forked inside the first sample window (75 frames), so 120
     * covers the whole contested region with margin and bounds the log.
     */
    private static final int EARLY_WINDOW = 120;

    /** Ticks seen since the match-start reset; timestamps the spend trace. */
    private static int spendTicks;

    /** Last credit balance per AI team id, for the spend trigger. */
    private static final java.util.Map<Integer, Long> lastCredits =
            new java.util.HashMap<Integer, Long>();

    private AiCadence() {
    }

    /**
     * Zeroes the spend trace at match start, beside the other instrument
     * resets ({@link MatchSetup}): the menu demo's AI spends too, and a
     * last-balance carried across the boundary would report the seed
     * application itself as a spend.
     */
    static void reset() {
        spendTicks = 0;
        lastCredits.clear();
    }

    /**
     * The current match tick, for instruments that stamp their lines with
     * it ({@link ThinkCount#charge}) -- the charge and spend traces
     * correlate by this number and nothing else.
     */
    static int tick() {
        return spendTicks;
    }

    /**
     * Logs the AI's task-aim state for one tick of the early window.
     *
     * <p>Rides {@link TickBracket}'s ENTER, which the engine drains at the
     * top of every tick body -- per-FRAME resolution where the sample line
     * has only per-75-frames. The per-sample probes proved the state layer
     * identical across forking twins while exactly one task re-aim ran in
     * one twin and not the other; only a per-tick timestamp of the aim
     * fields can say WHICH tick diverged and what the aim changed from and
     * to. Reads only, armed only with the draw tap, bounded to
     * {@link #EARLY_WINDOW} lines: a diagnostic that ran forever would bloat
     * every log for a window that ends two seconds in.
     */
    static void onTick() {
        if (!RandomTap.requested()) {
            return;
        }
        traceSpends();
        int tick = earlyTicks;
        if (tick >= EARLY_WINDOW) {
            return;
        }
        earlyTicks = tick + 1;
        Object engine = EngineHandle.current();
        if (engine == null) {
            return;
        }
        StringBuilder out = new StringBuilder();
        // The ledger's per-stream states and draw counts, per TICK where the
        // sample line has them per 75: the tick where a count steps in one
        // twin and not the other IS the divergent event, and whatever runs
        // on that tick is its mechanism -- the pinpoint the sample cadence
        // structurally cannot give.
        out.append(' ').append(RandomLedger.describe());
        // The think-entry counts, cumulative: the tick where twins' counts
        // step apart names the scheduler event no draw or state read could
        // see -- a think that ran and found nothing (ThinkCount).
        out.append(" think=").append(ThinkCount.describe());
        // The pre-tick queue's pending runnables, by class. The one-per-call
        // drain at the engine's i.b:2271 executes whatever a producer thread
        // managed to enqueue before this tick's boundary -- an OS race the
        // pinned delta cannot reach -- and the math draws the tap attributes
        // to that line happen INSIDE those runnables. Logging the queue's
        // contents per tick names the racer by its class the moment it
        // arrives (wiki log 2026-09-06, the pinpoint entry).
        Object queue = EngineAccess.readField(engine, EngineNames.PRETICK_QUEUE);
        java.util.Collection<?> pending = queue == null ? null : ObjectView.containedValues(queue);
        out.append(" k=[");
        if (pending != null) {
            boolean first = true;
            for (Object runnable : pending) {
                if (runnable == null) {
                    continue;
                }
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append(runnable.getClass().getName());
            }
        }
        out.append(']');
        Class<?> teams = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Class<?> ai = EngineAccess.pinnedClass(EngineNames.AI_CLASS);
        java.lang.reflect.Method lookup =
                EngineAccess.pinnedMethod(teams, EngineNames.TEAM_LOOKUP, int.class);
        int count = EngineAccess.readStaticIntField(teams, EngineNames.TEAM_COUNT);
        for (int index = 0; index < count; index++) {
            Object team = EngineAccess.invoke(lookup, null, Integer.valueOf(index));
            if (team == null || !ai.isInstance(team)) {
                continue;
            }
            Object held = EngineAccess.readField(team, EngineNames.AI_SUBCONTROLLERS);
            java.util.Collection<?> roster = held == null ? null : ObjectView.containedValues(held);
            out.append(" team=")
                    .append(EngineAccess.readIntField(team, EngineNames.TEAM_ID))
                    // Credits per tick, because the traced pair's FIRST
                    // divergence is the rival's worth moving 100 credits one
                    // sample apart with every stream still in lockstep -- a
                    // deterministic build decision on an input that differs.
                    // The tick where the balances part, and by how much,
                    // names the purchase (wiki log 2026-09-07).
                    .append(" cr=")
                    .append((long) EngineAccess.readDoubleField(team, EngineNames.CREDITS))
                    .append(" bm=")
                    .append(roster == null ? "?" : Integer.valueOf(roster.size()));
            if (roster == null) {
                continue;
            }
            for (Object controller : roster) {
                if (controller == null) {
                    continue;
                }
                if (EngineNames.AI_GROUP_CLASS.equals(controller.getClass().getName())) {
                    // Per-TICK group clocks, because the sample cadence
                    // measured them identical straight through a fork that
                    // clearly ran one more build-think in one twin: a clock
                    // that hits zero one tick apart and re-arms to the same
                    // value is invisible at 75-frame resolution and decisive
                    // at this one (wiki log 2026-09-07).
                    out.append(" i{e=").append(fieldValueOrAbsent(controller, "e"))
                            .append(",g=").append(fieldValueOrAbsent(controller, "g"))
                            .append(",i=").append(fieldValueOrAbsent(controller, "i"))
                            .append(",j=").append(fieldValueOrAbsent(controller, "j"))
                            .append('}');
                    continue;
                }
                if (!EngineNames.AI_TASK_CLASS.equals(controller.getClass().getName())) {
                    continue;
                }
                out.append(" n#")
                        .append(fieldValueOrAbsent(controller, "Q"))
                        .append(" S=")
                        .append(fieldValueOrAbsent(controller, "S"))
                        .append(",")
                        .append(fieldValueOrAbsent(controller, "T"))
                        .append(" d=")
                        .append(taskChoice(controller));
            }
        }
        Log.info("aitick t=" + tick + out);
    }

    /**
     * Prints one line per AI-team credit DECREASE, every tick, unbounded --
     * the seam the pinned full-length pair left standing: at frame 48,825
     * one twin held 972 credits and the other 72 with all three stream
     * hashes and every think counter still identical, a ~900-credit
     * draw-free spend the 75-frame cadence can bracket but never timestamp
     * (wiki log 2026-09-11). Income is continuous and spends are events,
     * so triggering on decrease keeps the trace to the events that matter.
     * Two runs' spend lines, diffed, name the first divergent purchase by
     * tick and amount.
     */
    private static void traceSpends() {
        Object engine = EngineHandle.current();
        if (engine == null) {
            return;
        }
        int tick = spendTicks;
        spendTicks = tick + 1;
        Class<?> teams = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Class<?> ai = EngineAccess.pinnedClass(EngineNames.AI_CLASS);
        java.lang.reflect.Method lookup =
                EngineAccess.pinnedMethod(teams, EngineNames.TEAM_LOOKUP, int.class);
        int count = EngineAccess.readStaticIntField(teams, EngineNames.TEAM_COUNT);
        for (int index = 0; index < count; index++) {
            Object team = EngineAccess.invoke(lookup, null, Integer.valueOf(index));
            if (team == null || !ai.isInstance(team)) {
                continue;
            }
            int id = EngineAccess.readIntField(team, EngineNames.TEAM_ID);
            double raw = EngineAccess.readDoubleField(team, EngineNames.CREDITS);
            long credits = (long) raw;
            Long last = lastCredits.put(Integer.valueOf(id), Long.valueOf(credits));
            if (last != null && credits < last.longValue()) {
                // The raw double rides along because every fork so far
                // strikes from the SAME truncated balance at an
                // affordability edge -- a fractional divergence would be
                // invisible to the integer view and decisive here
                // (wiki log 2026-09-11).
                Log.info(
                        "credspend t=" + tick + " team=" + id + " " + last + "->" + credits
                                + " raw=" + raw);
            }
        }
    }

    /**
     * Renders a task's chosen group as class:category, or its plain value
     * when it is not a roster object -- the task's {@code d} holds either a
     * chosen group or null, and which group it holds is the fact the tick
     * line exists to timestamp.
     *
     * @param task The n-task.
     * @return A compact signature of the choice.
     */
    private static String taskChoice(Object task) {
        java.lang.reflect.Field field;
        try {
            field = task.getClass().getDeclaredField("d");
        } catch (NoSuchFieldException absent) {
            return "absent";
        }
        try {
            field.setAccessible(true);
            Object chosen = field.get(task);
            if (chosen == null) {
                return "null";
            }
            return chosen.getClass().getSimpleName() + ':'
                    + fieldValueOrAbsent(chosen, "b") + '/' + fieldValueOrAbsent(chosen, "c");
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read task choice on " + task.getClass().getName(), e);
        }
    }

    /**
     * Renders the cadence state, one compact line body.
     *
     * <p>Runs on the game thread at each sample boundary, beside the ledger
     * and tap lines it exists to explain. Reads are through the pinned-name
     * machinery and throw when the layout moved — a diagnostic run that
     * silently recorded the wrong field would be worse than one that
     * crashed naming it.
     *
     * @param engine The live engine instance.
     * @return The line body, without prefix or frame.
     */
    static String describe(Object engine) {
        StringBuilder out = new StringBuilder();
        // Per-sample think counts, so cluster traces carry the discriminant
        // past the 120-tick early window: two twins' counts at the same
        // sample either match or name the window where the schedule forked.
        out.append("think=").append(ThinkCount.describe()).append(' ');
        Object map = EngineAccess.readField(engine, EngineNames.MAP);
        Object pools = map == null ? null : EngineAccess.readField(map, EngineNames.MAP_POOL_POINTS);
        java.util.Collection<?> poolList = pools == null ? null : ObjectView.containedValues(pools);
        out.append("pools=").append(poolList == null ? "?" : Integer.valueOf(poolList.size()));

        Class<?> teams = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Class<?> ai = EngineAccess.pinnedClass(EngineNames.AI_CLASS);
        java.lang.reflect.Method lookup =
                EngineAccess.pinnedMethod(teams, EngineNames.TEAM_LOOKUP, int.class);
        int count = EngineAccess.readStaticIntField(teams, EngineNames.TEAM_COUNT);
        for (int index = 0; index < count; index++) {
            Object team = EngineAccess.invoke(lookup, null, Integer.valueOf(index));
            if (team == null || !ai.isInstance(team)) {
                continue;
            }
            Object held = EngineAccess.readField(team, EngineNames.AI_SUBCONTROLLERS);
            java.util.Collection<?> roster = held == null ? null : ObjectView.containedValues(held);
            out.append(" | team=")
                    .append(EngineAccess.readIntField(team, EngineNames.TEAM_ID))
                    .append(" cr=")
                    .append((long) EngineAccess.readDoubleField(team, EngineNames.CREDITS))
                    .append(" aT=")
                    .append(EngineAccess.readFloatField(team, EngineNames.AI_CLOCK_BASE_GROUP))
                    .append(" aU=")
                    .append(EngineAccess.readFloatField(team, EngineNames.AI_CLOCK_UNIT_GROUP))
                    .append(" bm=")
                    .append(roster == null ? "?" : Integer.valueOf(roster.size()));
            if (roster != null) {
                appendRoster(out, roster);
            }
        }
        return out.toString();
    }

    /**
     * Appends the roster's composition, because a COUNT that matches across
     * twins can hide a composition that does not: the E/F pair agreed on
     * bm=4 through the exact window where one twin consulted the task
     * chooser's random gate once more than the other, so which classes and
     * which group categories those four are is the next discriminant.
     *
     * <p>Rendered by simple class name plus, for objects carrying them, the
     * {@code b}/{@code c} category enums the gate filters on -- generically
     * via the field walk, like {@link AiZones}, so the reading cannot bake
     * in the interpretation it exists to test.
     *
     * @param out Line being built.
     * @param roster The team's sub-controllers.
     */
    private static void appendRoster(StringBuilder out, java.util.Collection<?> roster) {
        out.append(" [");
        boolean first = true;
        for (Object controller : roster) {
            if (controller == null) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            out.append(controller.getClass().getSimpleName());
            String category = fieldValueOrAbsent(controller, "b");
            String kind = fieldValueOrAbsent(controller, "c");
            if (category != null) {
                out.append(':').append(category);
            }
            if (kind != null) {
                out.append('/').append(kind);
            }
            if (EngineNames.AI_GROUP_CLASS.equals(controller.getClass().getName())) {
                // The group's own cadence clocks, the six the reset table
                // names. The clean pair forks on whether a group's
                // build-a-building path runs at tick 33 (i.g -> o.e, the
                // placement chooser), and these clocks are what schedule
                // that think -- twin trajectories either drift here, naming
                // the clock, or match and push the seam deeper (wiki log
                // 2026-09-07).
                out.append("{e=").append(fieldValueOrAbsent(controller, "e"))
                        .append(" g=").append(fieldValueOrAbsent(controller, "g"))
                        .append(" i=").append(fieldValueOrAbsent(controller, "i"))
                        .append(" j=").append(fieldValueOrAbsent(controller, "j"))
                        .append(" k=").append(fieldValueOrAbsent(controller, "k"))
                        .append(" m=").append(fieldValueOrAbsent(controller, "m"))
                        .append(" q=").append(memberFingerprint(controller))
                        .append('}');
            }
        }
        out.append(']');
    }

    /**
     * Renders a group's unit-membership fingerprint --
     * {@code size:sum-of-entity-ids} over the group roster {@code q} the
     * builder-finder iterates. Order-independent by construction, because
     * the question it answers is MEMBERSHIP: the tm pair's twins entered
     * tick 5,853 with balances identical to the double's last bit, tried
     * identical candidates, and one twin's finder answered null -- and
     * {@code q}'s backing store is an insertion-ordered array, so the
     * remaining suspect is who is IN it (wiki log 2026-09-11). No hash of
     * any prior instrument covers this state.
     *
     * @param group The a.i group.
     * @return {@code <size>:<id-sum>}, or {@code ?} when the roster is
     *     unreadable.
     */
    private static String memberFingerprint(Object group) {
        Object held = EngineAccess.readField(group, "q");
        java.util.Collection<?> members = held == null ? null : ObjectView.containedValues(held);
        if (members == null) {
            return "?";
        }
        long sum = 0;
        int count = 0;
        for (Object unit : members) {
            if (unit == null) {
                continue;
            }
            count++;
            sum += EngineAccess.readLongField(unit, EngineNames.ENTITY_ID);
        }
        return count + ":" + sum;
    }

    /**
     * Renders one declared field's value, or null when the class declares no
     * such field -- absence is a real answer here, not an error: the roster
     * mixes classes and only some carry the category enums, and the probe's
     * point is to record which.
     *
     * @param target The controller.
     * @param name The field to look for on the controller's own class.
     * @return The value rendered via its own {@code toString}, {@code "null"}
     *     for a null value, or null when the field is not declared.
     * @throws IllegalStateException When the field exists but cannot be read,
     *     which means the access machinery moved, not the layout.
     */
    private static String fieldValueOrAbsent(Object target, String name) {
        java.lang.reflect.Field field;
        try {
            field = target.getClass().getDeclaredField(name);
        } catch (NoSuchFieldException absent) {
            return null;
        }
        try {
            field.setAccessible(true);
            Object value = field.get(target);
            return value == null ? "null" : value.toString();
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read declared field " + name + " on "
                            + target.getClass().getName(),
                    e);
        }
    }
}
