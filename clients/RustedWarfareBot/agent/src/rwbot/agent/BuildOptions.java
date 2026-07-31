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
     * <p>Carries the engine's interned key name alongside the produced type
     * because the two answer different questions. The type is what a plan
     * names; the key is what addresses one action among a unit's several --
     * including the ones that concern no type at all, which nothing but the
     * key can name.
     */
    static final class Option {

        private final long unitId;
        private final String produces;
        private final String key;
        private final boolean placed;
        private final boolean available;
        private final boolean makesSomething;
        private final int price;

        Option(
                long unitId,
                String produces,
                String key,
                boolean placed,
                boolean available,
                boolean makesSomething,
                int price) {
            this.unitId = unitId;
            this.produces = produces;
            this.key = key;
            this.placed = placed;
            this.available = available;
            this.makesSomething = makesSomething;
            this.price = price;
        }

        long unitId() {
            return unitId;
        }

        String produces() {
            return produces;
        }

        /**
         * The engine's interned key name for the action.
         *
         * <p>The dispatch handle. The engine also exposes a per-action index,
         * and it is not a selector: every action on a unit answers the same
         * figure, so four matches running the "unlock" it dispatched was the
         * first action on the list -- the rally point. The key is what the
         * engine's own executor resolves actions by, so it is what the wire
         * carries (wiki: mechanics-build-actions).
         */
        String key() {
            return key;
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

        /**
         * Whether the engine calls this an action that makes something.
         *
         * <p>Published rather than filtered on, and that distinction cost a
         * whole category of the game. This class used to drop any action that
         * neither placed something nor answered true here, on the reading that
         * the rest were stops and rallies. An upgrade is neither: the asset
         * declares it as {@code convertTo}, and the engine files conversions in
         * a separate list from build actions before wrapping them back into the
         * one the agent reads. Opponents are observed holding upgraded
         * extractors and upgraded turrets while ours publish no options at all
         * (wiki: policy-holding-ground).
         *
         * <p>Whether an action is worth taking is a decision, and decisions
         * belong to the planner. The agent's job is to say what the engine
         * offers.
         */
        boolean makesSomething() {
            return makesSomething;
        }

        /**
         * What the action costs in credits.
         *
         * <p>The engine's own figure, read from the accessor every action
         * implements. It is what tells a factory's tier upgrade apart from
         * its rally point -- the only two readings of an action that
         * concerns no type -- and the planner's budget claims the same
         * number the engine will charge (wiki: mechanics-build-actions).
         */
        int price() {
            return price;
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
        Method available =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_AVAILABLE, entityClass);
        Method locked =
                EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_LOCKED, entityClass);
        Method price = EngineAccess.pinnedMethod(actionClass, EngineNames.ACTION_PRICE);

        for (Object unit : Perception.ownedUnits(engine)) {
            long id = Perception.idOf(unit);
            for (Object action : actionsOf(actions, unit)) {
                if (action == null) {
                    continue;
                }
                // Every action is published, including the ones that concern no
                // type at all. Two filters used to live here -- drop an action
                // whose type is null, and drop one that neither places nor
                // "makes something" -- and together they hid upgrades, which
                // the engine models as conversions rather than as builds. The
                // planner can ignore an action; it cannot ignore one it was
                // never told about (wiki: policy-holding-ground).
                Object type = EngineAccess.invoke(makes, action);
                options.add(
                        new Option(
                                id,
                                type == null ? "" : Perception.nameOfType(type),
                                keyNameOf(action),
                                EngineAccess.invoke(placedType, action) != null,
                                isUsable(action, unit, available, locked),
                                Boolean.TRUE.equals(
                                        EngineAccess.invoke(makesSomething, action)),
                                intOf(
                                        EngineAccess.invoke(price, action),
                                        EngineNames.ACTION_PRICE)));
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
        // Matched on the type alone. This used to skip any action the engine
        // did not call "makes something", which is the same filter the listing
        // path applied -- so removing it there and leaving it here produced the
        // worst of both: the planner was offered an upgrade it could see and
        // then the dispatch could not find it, and the agent threw inside the
        // engine's script thread and crashed the game. An action naming the
        // wanted type IS the action, whatever the engine calls its category
        // (wiki: policy-holding-ground).
        for (Object action : actionsOf(actions, unit)) {
            if (action == null) {
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
     * Finds a unit's action by the engine's interned key name.
     *
     * <p>The ability path's lookup: the option stream publishes every action
     * with its key, and an ability order fires that key back. Matching on
     * the same string the listing published is what keeps the two from
     * drifting. It matched on the engine's per-action index before, and the
     * index is not a selector -- every action on a unit answers the same
     * figure, so the dispatch always resolved the first action on the list,
     * which is the rally point (wiki: mechanics-build-actions).
     *
     * @param unit The unit to search.
     * @param key The action's key name, from the option stream.
     * @return The action, or null when the unit has none under that key.
     */
    static Object actionByKey(Object unit, String key) {
        Class<?> entityClass = EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS);
        Method actions = EngineAccess.pinnedMethod(entityClass, EngineNames.ACTIONS);
        for (Object action : actionsOf(actions, unit)) {
            if (action == null) {
                continue;
            }
            if (key.equals(keyNameOf(action))) {
                return action;
            }
        }
        return null;
    }

    /**
     * Reads an action's interned key name.
     *
     * <p>The engine keys every action with an interned object whose name is
     * the stable identifier its own executor resolves by ({@code u_builder},
     * {@code c_1}, ...). An action with no key, or a key with no name, reads
     * as empty -- such an action cannot be dispatched, and publishing the
     * blank is what lets the planner see that rather than invent it.
     *
     * @param action The action.
     * @return The key's name, or empty when it has none.
     */
    static String keyNameOf(Object action) {
        Object key =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(action.getClass(), EngineNames.ACTION_KEY),
                        action);
        if (key == null) {
            return "";
        }
        Object name =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(key.getClass(), EngineNames.ACTION_KEY_NAME),
                        key);
        return name instanceof String ? (String) name : "";
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
        StringBuilder out = new StringBuilder();
        // Listed on the same rule the lookup uses, so the failure message and
        // the lookup can never disagree. When they did, an extractor that was
        // offering an upgrade was reported as able to "make nothing".
        for (Object action : actionsOf(actions, unit)) {
            if (action == null) {
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
