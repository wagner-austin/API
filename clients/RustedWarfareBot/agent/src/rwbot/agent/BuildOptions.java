package rwbot.agent;

import java.lang.reflect.Method;

/**
 * What each unit the player owns can actually make.
 *
 * <p>Nothing the bot already reads answers this. The stat catalogue prices
 * every unit and says nothing about who produces it; the Land Factory's own
 * catalogue entry offers the sentence "Builds land units", which is prose. The
 * cost of guessing has been paid twice — once ordering a builder to construct a
 * laboratory it had no action for, and once ordering a factory to produce three
 * plausible-sounding tank names, none of which it makes.
 *
 * <p>So it is enumerated from the engine. Every unit carries a list of actions;
 * a builder's produce buildings and a factory's produce units, and the order
 * path resolves both through the same lookup. Reading that list is reading the
 * exact set the engine will accept (wiki: mechanics-build-actions).
 *
 * <p><b>Read per entity rather than per type.</b> The action list is declared
 * on the entity base class and overridden per unit class, so it is not
 * reachable from the type registry the way a placement flag is. Availability is
 * also genuinely per entity — the same predicate the engine consults takes the
 * unit as its argument — so a static table could not express it anyway.
 */
final class BuildOptions {

    /**
     * One thing a unit can make.
     *
     * <p>Carries the selector index alongside the produced type because the two
     * answer different questions. The type is what a plan names; the index is
     * what disambiguates when a unit has more than one action producing the
     * same type, which is what the build command's integer argument selects.
     */
    static final class Option {

        private final long unitId;
        private final String produces;
        private final int actionIndex;
        private final boolean placed;
        private final boolean available;

        Option(
                long unitId,
                String produces,
                int actionIndex,
                boolean placed,
                boolean available) {
            this.unitId = unitId;
            this.produces = produces;
            this.actionIndex = actionIndex;
            this.placed = placed;
            this.available = available;
        }

        long unitId() {
            return unitId;
        }

        String produces() {
            return produces;
        }

        int actionIndex() {
            return actionIndex;
        }

        /**
         * Whether this thing is placed at a chosen position.
         *
         * <p>Decides which verb orders it. A structure is placed, and goes
         * through the build-waypoint path; a unit rolls out of the building
         * that made it, and goes through the action path.
         */
        boolean placed() {
            return placed;
        }

        boolean available() {
            return available;
        }
    }

    private BuildOptions() {
    }

    /**
     * Lists everything the current player's units can make.
     *
     * <p>Only owned units are enumerated. An opponent's factory is visible and
     * its actions are readable, but the bot cannot order it, so reporting them
     * would be information with no legitimate use.
     *
     * <p>Two disjoint families, and both are wanted. A structure-placing action
     * reports the type it would place and answers the engine's "makes
     * something" predicate with false; a unit-producing action places nothing
     * and answers it with true. Neither test alone finds both — filtering on
     * the placement accessor loses every factory, filtering on the predicate
     * loses every builder, and this code made each mistake in turn before
     * settling on their union.
     *
     * <p>Everything else is skipped. Most of a unit's actions are not
     * construction at all — stop, guard, set a rally point — and they fail both
     * tests, so the agent has to recognise no names.
     *
     * <p>Must run on the game thread.
     *
     * @param engine The live engine instance.
     * @return One option per producible type per owned unit, in roster order.
     */
    static java.util.List<Option> ownedOptions(Object engine) {
        java.util.List<Option> options = new java.util.ArrayList<Option>();
        Class<?> entityClass = EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS);
        Class<?> actionClass = EngineAccess.pinnedClass(EngineNames.ACTION_CLASS);
        Method actions = EngineAccess.pinnedMethod(entityClass, EngineNames.ACTIONS);
        Method makes = EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_MAKES);
        Method placedType =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_PLACED_TYPE);
        Method makesSomething =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_MAKES_SOMETHING);
        Method index = EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_INDEX);
        Method available =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_AVAILABLE, entityClass);
        Method locked =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_LOCKED, entityClass);

        for (Object unit : Perception.ownedUnits(engine)) {
            long id = Perception.idOf(unit);
            for (Object action : actionsOf(actions, unit)) {
                if (action == null) {
                    continue;
                }
                Object type = EngineAccess.invoke(makes, action);
                if (type == null) {
                    continue;
                }
                boolean placed = EngineAccess.invoke(placedType, action) != null;
                if (!placed && !Boolean.TRUE.equals(EngineAccess.invoke(makesSomething, action))) {
                    continue;
                }
                options.add(
                        new Option(
                                id,
                                Perception.nameOfType(type),
                                intOf(
                                        EngineAccess.invoke(index, action),
                                        EngineNames.ACTION_INDEX),
                                placed,
                                isUsable(action, unit, available, locked)));
            }
        }
        return options;
    }

    /**
     * Finds the action by which a unit makes a given type.
     *
     * <p>Matches on the type the action makes rather than on any name the agent
     * composes, so the caller never has to know how the engine keys its
     * actions.
     *
     * @param unit The unit to search.
     * @param typeName The type wanted.
     * @return The action, or null when the unit has none making that type.
     */
    static Object actionMaking(Object unit, String typeName) {
        Class<?> entityClass = EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS);
        Class<?> actionClass = EngineAccess.pinnedClass(EngineNames.ACTION_CLASS);
        Method actions = EngineAccess.pinnedMethod(entityClass, EngineNames.ACTIONS);
        Method makes = EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_MAKES);
        Method makesSomething =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_MAKES_SOMETHING);
        for (Object action : actionsOf(actions, unit)) {
            if (action == null
                    || !Boolean.TRUE.equals(EngineAccess.invoke(makesSomething, action))) {
                continue;
            }
            Object type = EngineAccess.invoke(makes, action);
            if (type != null && typeName.equals(Perception.nameOfType(type))) {
                return action;
            }
        }
        return null;
    }

    /**
     * Lists what a unit can make, for a failure message.
     *
     * <p>An order refused for naming something the subject cannot make is the
     * most common way to get this wrong, and the useful half of that error is
     * what it <em>could</em> have been asked for.
     *
     * @param unit The unit to describe.
     * @return A comma-separated list, or a note that it makes nothing.
     */
    static String describeMakeable(Object unit) {
        Class<?> entityClass = EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS);
        Class<?> actionClass = EngineAccess.pinnedClass(EngineNames.ACTION_CLASS);
        Method actions = EngineAccess.pinnedMethod(entityClass, EngineNames.ACTIONS);
        Method makes = EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_MAKES);
        Method makesSomething =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_MAKES_SOMETHING);
        StringBuilder out = new StringBuilder();
        for (Object action : actionsOf(actions, unit)) {
            if (action == null
                    || !Boolean.TRUE.equals(EngineAccess.invoke(makesSomething, action))) {
                continue;
            }
            Object type = EngineAccess.invoke(makes, action);
            if (type == null) {
                continue;
            }
            if (out.length() > 0) {
                out.append(", ");
            }
            out.append(Perception.nameOfType(type));
        }
        return out.length() == 0 ? "nothing" : out.toString();
    }

    /**
     * Reports whether a unit may use one of its actions right now.
     *
     * <p>Two predicates, and both have to pass. An action can be present and
     * unavailable — insufficient tech, a prerequisite building missing — and it
     * can be separately locked. The engine checks both before accepting a build
     * waypoint, so anything less here would report an option the order path
     * would then refuse.
     *
     * @param action The action.
     * @param unit The unit that owns it.
     * @param available The availability predicate.
     * @param locked The lock predicate.
     * @return True when the action is available and not locked.
     */
    private static boolean isUsable(Object action, Object unit, Method available, Method locked) {
        return Boolean.TRUE.equals(EngineAccess.invoke(available, action, unit))
                && !Boolean.TRUE.equals(EngineAccess.invoke(locked, action, unit));
    }

    /**
     * The engine's answers to every gate a production order can be stopped by
     * that can be asked without changing the game.
     *
     * <p>That qualification is the whole design. The engine's enqueue path
     * applies five conditions in order — the action resolves from its key, it
     * is available, it applies, the player is under the unit cap, and the cost
     * is paid — and the last one is not a question. It is
     * {@code check-then-charge}: asking it deducts the credits. So a diagnostic
     * can report the first four and must not touch the fifth, and an order that
     * passes all four can still be refused at the till.
     */
    static final class Gates {

        private final boolean applies;
        private final boolean available;
        private final boolean locked;
        private final boolean room;

        Gates(boolean applies, boolean available, boolean locked, boolean room) {
            this.applies = applies;
            this.available = available;
            this.locked = locked;
            this.room = room;
        }

        /**
         * Names the first gate that would stop this order, or null when none
         * would.
         *
         * <p>Ordered by how specific the answer is rather than by how the
         * engine evaluates them. {@code applies} folds in the lock, a cooldown
         * and affordability, so reporting it when the narrower {@code locked}
         * already explains the refusal would name a symptom over its cause.
         *
         * @return The gate's name, or null when all four are open.
         */
        String closed() {
            if (!available) {
                return "available=false: the engine skips the action outright";
            }
            if (locked) {
                return "locked=true: the unit has this action locked";
            }
            if (!applies) {
                return "applies=false: locked, on cooldown, or unaffordable";
            }
            if (!room) {
                return "room=false: the player is at the unit cap";
            }
            return null;
        }

        @Override
        public String toString() {
            return "[applies=" + applies
                    + " available=" + available
                    + " locked=" + locked
                    + " room=" + room
                    + "]";
        }
    }

    /**
     * Reads every gate the engine applies before queueing an action, except the
     * one that charges for it.
     *
     * <p>This exists because the refusal is silent. The engine's own two
     * complaints on this path — the action not resolving, and it not being
     * available — go through a logger with a static counter that is never
     * reset, so it prints four messages per process and then nothing, for the
     * rest of the run. Everything past that point returns null without a word.
     *
     * <p>Reading the gates costs nothing and changes nothing: all four are pure
     * reads of the engine's state, and the one that would not be is deliberately
     * absent (wiki: mechanics-build-actions).
     *
     * @param action The action.
     * @param unit The unit that would run it.
     * @return The four readable gates.
     */
    static Gates gatesOf(Object action, Object unit) {
        Class<?> entityClass = EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS);
        Class<?> actionClass = EngineAccess.pinnedClass(EngineNames.ACTION_CLASS);
        // False, always. The true branch routes affordability through the
        // engine's check-and-charge helper and spends the credits.
        boolean applies =
                Boolean.TRUE.equals(
                        EngineAccess.invoke(
                                EngineAccess.pinnedMethod(
                                        actionClass,
                                        EngineNames.ACTION_APPLIES,
                                        entityClass,
                                        boolean.class),
                                action,
                                unit,
                                Boolean.FALSE));
        boolean available =
                Boolean.TRUE.equals(
                        EngineAccess.invoke(
                                EngineAccess.pinnedMethod(
                                        actionClass, EngineNames.ACTION_AVAILABLE, entityClass),
                                action,
                                unit));
        boolean locked =
                Boolean.TRUE.equals(
                        EngineAccess.invoke(
                                EngineAccess.pinnedMethod(
                                        actionClass, EngineNames.ACTION_LOCKED, entityClass),
                                action,
                                unit));
        return new Gates(applies, available, locked, hasRoomFor(action, unit));
    }

    /**
     * Reports whether the player may hold one more unit.
     *
     * <p>The cap counts units excluding buildings and including ones already
     * queued, so a factory with a full queue is at the cap before any of it has
     * rolled out. An action that makes nothing is not capped at all, which is
     * the engine's own short-circuit and not an exemption invented here.
     *
     * @param action The action.
     * @param unit The unit that would run it.
     * @return True when the action is uncapped or the player is under the cap.
     */
    private static boolean hasRoomFor(Object action, Object unit) {
        Object makesSomething =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(
                                EngineAccess.pinnedClass(EngineNames.ACTION_CLASS),
                                EngineNames.ACTION_MAKES_SOMETHING),
                        action);
        if (!Boolean.TRUE.equals(makesSomething)) {
            return true;
        }
        // Not guarded against a null owner. Every caller reaches this through a
        // unit the local player owns, so an unowned one is a broken invariant
        // rather than a state to accommodate -- and reporting "room=true" for it
        // would put a reassuring number in a diagnostic that exists precisely
        // because reassuring numbers are how this path fails.
        Object owner = EngineAccess.readField(unit, EngineNames.OWNER);
        Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        int held =
                intOf(
                        EngineAccess.invoke(
                                EngineAccess.pinnedMethod(teamClass, EngineNames.TEAM_UNIT_COUNT),
                                owner),
                        EngineNames.TEAM_UNIT_COUNT);
        int cap =
                intOf(
                        EngineAccess.invoke(
                                EngineAccess.pinnedMethod(teamClass, EngineNames.TEAM_UNIT_CAP),
                                owner),
                        EngineNames.TEAM_UNIT_CAP);
        return held < cap;
    }

    /** Reads a unit's action list, treating an absent one as empty. */
    private static Iterable<?> actionsOf(Method actions, Object unit) {
        Object value = EngineAccess.invoke(actions, unit);
        if (value == null) {
            return java.util.Collections.emptyList();
        }
        if (!(value instanceof Iterable)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.ACTIONS + "() on " + unit.getClass().getName()
                            + " is not iterable" + EngineNames.PIN);
        }
        return (Iterable<?>) value;
    }

    /**
     * Unwraps a reflected {@code int} return.
     *
     * @param value The reflected return.
     * @param accessor The accessor that produced it, for the failure message.
     * @return The int.
     */
    private static int intOf(Object value, String accessor) {
        if (!(value instanceof Integer)) {
            throw new IllegalStateException(
                    "rw-agent: " + accessor + "() did not return an int" + EngineNames.PIN);
        }
        return ((Integer) value).intValue();
    }
}
