package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * What the bot can see: the roster, positions, health, ownership and economy.
 *
 * <p>Reads only. Nothing here writes to the simulation and nothing here decides
 * anything — selection is the planner's job (wiki:
 * runtime-split-java-agent-python-brain). Dispatch is {@link Orders}; the
 * obfuscated names and the reflection are {@link EngineBindings}.
 *
 * <p><b>Visibility is delegated, not reimplemented.</b> The master entity list
 * holds every unit on the map, so enumerating it would give the bot perfect
 * information. {@link #visibleEntities} asks the engine's own per-player fog
 * test instead, which is what keeps perception legitimate (wiki:
 * perception-visibility).
 */
final class Perception {

    private Perception() {
    }

    /**
     * Lists every order-taking entity the engine's current player owns.
     *
     * <p>Trees are EngineAccess.entities too, so the tree subclass is excluded explicitly
     * rather than by hoping the first hit is a unit -- a size coincidence
     * between a sprite list and the unit count has already produced one wrong
     * answer on this codebase (wiki: engine-entity-model).
     *
     * <p>Buildings are deliberately kept. They take orders too (rally points,
     * build queues), and filtering them here would put a mobility judgement in
     * the dispatch layer. The roster is reported in list order so a caller can
     * address a unit by index.
     *
     * @param engine The live engine instance.
     * @return The owned EngineAccess.entities, in entity-list order. Empty when the player
     *     owns nothing or there is no current player.
     */
    static java.util.List<Object> ownedUnits(Object engine) {
        java.util.List<Object> owned = new java.util.ArrayList<Object>();
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return owned;
        }
        Class<?> treeClass = EngineAccess.pinnedClass(EngineNames.TREE_CLASS);
        Class<?> orderableClass = EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS);
        for (Object entity : EngineAccess.entities()) {
            if (entity == null || treeClass.isInstance(entity) || !orderableClass.isInstance(entity)) {
                continue;
            }
            if (EngineAccess.readField(entity, EngineNames.OWNER) == team) {
                owned.add(entity);
            }
        }
        return owned;
    }

    /**
     * Lists every entity the current player can legitimately see.
     *
     * <p>Own units plus enemy units the engine reports as visible, judged by
     * the engine's own fog test rather than by reading the master list
     * directly. That distinction is the whole point: {@code am.bE} holds every
     * unit on the map, so a bot enumerating it would have perfect information
     * and would no longer be playing the game a human plays (wiki:
     * multiplayer-portability-invariants).
     *
     * <p>Trees are excluded, as in {@link #ownedUnits}. Buildings are kept:
     * they are legitimate targets and legitimate order-takers, and filtering
     * them here would put a judgement in the perception layer.
     *
     * @param engine The live engine instance.
     * @return The visible EngineAccess.entities, in entity-list order. Empty when there is
     *     no current player.
     */
    static java.util.List<Object> visibleEntities(Object engine) {
        java.util.List<Object> visible = new java.util.ArrayList<Object>();
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return visible;
        }
        Class<?> treeClass = EngineAccess.pinnedClass(EngineNames.TREE_CLASS);
        Class<?> orderableClass = EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS);
        Class<?> entityClass = EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS);
        Method visibleTo = EngineAccess.pinnedMethod(entityClass, EngineNames.VISIBLE_TO, EngineAccess.pinnedClass(EngineNames.TEAM_CLASS));
        for (Object entity : EngineAccess.entities()) {
            if (entity == null || treeClass.isInstance(entity) || !orderableClass.isInstance(entity)) {
                continue;
            }
            Object seen = EngineAccess.invoke(visibleTo, entity, team);
            if (Boolean.TRUE.equals(seen)) {
                visible.add(entity);
            }
        }
        return visible;
    }

    /**
     * Reports whether an entity belongs to the engine's current player.
     *
     * @param engine The live engine instance.
     * @param entity The entity to test.
     * @return True when the entity's owner is the current player.
     */
    static boolean isOwnedByLocalPlayer(Object engine, Object entity) {
        return EngineAccess.readField(entity, EngineNames.OWNER) == EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
    }

    /**
     * Reports whether an entity's owner is hostile to the current player.
     *
     * <p>Asked of the engine rather than derived from ownership. "Not mine" is
     * the wrong test twice over: an allied player's units are not mine and not
     * hostile, and neutral map objects are neither. The engine compares
     * alliance group and excludes the neutral team explicitly, and that is the
     * comparison used here (wiki: perception-visibility).
     *
     * <p>An entity with no owner is not hostile, which is the same answer the
     * engine gives for the neutral team.
     *
     * @param engine The live engine instance.
     * @param entity The entity to test.
     * @return True when the current player and the entity's owner are on
     *     opposing sides.
     */
    static boolean isHostileToLocalPlayer(Object engine, Object entity) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        Object owner = EngineAccess.readField(entity, EngineNames.OWNER);
        if (team == null || owner == null) {
            return false;
        }
        return isHostileBetween(team, owner);
    }

    /**
     * Reports whether one team counts the other as an enemy.
     *
     * <p>The engine's own alliance comparison, and deliberately not the negation
     * of "same team": an ally and the neutral team are both other teams and
     * neither is hostile. Shared by the per-entity test and the per-player
     * scoreboard so the two cannot come to different answers about the same
     * pair.
     *
     * @param team The team asking.
     * @param other The team asked about.
     * @return The engine's answer.
     * @throws IllegalStateException When the comparison does not answer.
     */
    static boolean isHostileBetween(Object team, Object other) {
        Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Object hostile =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(
                                teamClass, EngineNames.TEAM_HOSTILE_TO, teamClass),
                        team,
                        other);
        if (!(hostile instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TEAM_HOSTILE_TO + "() did not return a boolean"
                            + EngineNames.PIN);
        }
        return ((Boolean) hostile).booleanValue();
    }

    /**
     * Returns an entity's movement layer by name, e.g. {@code "LAND"}.
     *
     * <p>Which layer a unit travels on decides which terrain it can cross, and
     * the engine keeps a separate connectivity grid per layer. Reported as the
     * enum's own constant name rather than an ordinal, because the names
     * survived obfuscation and the field letters did not
     * (wiki: engine-name-oracle).
     *
     * @param entity The entity to read.
     * @return Its layer name.
     */
    static String movementOf(Object entity) {
        Object layer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(entity.getClass(), EngineNames.ENTITY_MOVEMENT),
                        entity);
        if (!(layer instanceof Enum)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.ENTITY_MOVEMENT + "() did not return a movement"
                            + " layer" + EngineNames.PIN);
        }
        return ((Enum<?>) layer).name();
    }

    /**
     * Returns the connectivity component an entity stands in, on its own layer.
     *
     * <p>Two things can reach each other exactly when their component ids match
     * and both are real. That is the engine's own reachability test reduced to
     * a comparison, which is why this is worth carrying rather than recomputing
     * (wiki: mechanics-movement-layers).
     *
     * @param entity The entity to locate.
     * @return Its component id, or a negative when the point has none.
     */
    static int pathGroupOf(Object entity) {
        float[] at = positionOf(entity);
        Object layer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(entity.getClass(), EngineNames.ENTITY_MOVEMENT),
                        entity);
        return pathGroupAt(at[0], at[1], layer);
    }

    /**
     * Returns the connectivity component of a world point on the land layer.
     *
     * <p>Land specifically, and the field it feeds is named for it. Every
     * builder in the base game travels on land, so this answers the question
     * the planner actually asks; a hover or naval builder would need its own
     * grid carried alongside, and naming the layer here keeps that gap visible
     * instead of letting a mismatched comparison look like an answer.
     *
     * @param x World x.
     * @param y World y.
     * @return The component id, or a negative when the point has none.
     */
    static int landPathGroupAt(float x, float y) {
        Object land = movementLayer(EngineNames.MOVEMENT_LAND);
        int here = pathGroupAt(x, y, land);
        if (here >= 0) {
            return here;
        }
        // A resource-pool tile is itself impassable -- every one on the
        // archived map reports -1 -- so the centre alone would call every pool
        // unreachable and stop the economy. What matters is whether a builder
        // can stand beside it, so the four neighbours are sampled. The engine's
        // own AI does the same thing for the same reason: its zone-reachability
        // check tries the centre and then four points around it before giving
        // up (wiki: mechanics-movement-layers).
        Object map = EngineAccess.readField(EngineHandle.current(), EngineNames.MAP);
        float step = EngineAccess.readIntField(map, EngineNames.TILE_WIDTH);
        float[][] neighbours = {{step, 0.0f}, {-step, 0.0f}, {0.0f, step}, {0.0f, -step}};
        for (float[] offset : neighbours) {
            int beside = pathGroupAt(x + offset[0], y + offset[1], land);
            if (beside >= 0) {
                return beside;
            }
        }
        // Genuinely nothing adjacent is walkable. Reported as the centre's own
        // answer rather than as a distinct code: every negative already means
        // "no component here", and inventing a sixth one would only give the
        // reader something else to interpret.
        return here;
    }

    /** Asks the engine for the component id of a point on one layer. */
    private static int pathGroupAt(float x, float y, Object layer) {
        Object group =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(
                                EngineAccess.pinnedClass(EngineNames.PATHING_CLASS),
                                EngineNames.PATH_GROUP_AT,
                                float.class,
                                float.class,
                                EngineAccess.pinnedClass(EngineNames.MOVEMENT_CLASS)),
                        null,
                        Float.valueOf(x),
                        Float.valueOf(y),
                        layer);
        if (!(group instanceof Short)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.PATH_GROUP_AT + "() did not return a short"
                            + EngineNames.PIN);
        }
        return ((Short) group).intValue();
    }

    /** Resolves one movement-layer constant by its engine name. */
    private static Object movementLayer(String name) {
        Object[] layers =
                EngineAccess.pinnedClass(EngineNames.MOVEMENT_CLASS).getEnumConstants();
        if (layers == null) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.MOVEMENT_CLASS + " is not an enum"
                            + EngineNames.PIN);
        }
        for (Object layer : layers) {
            if (name.equals(((Enum<?>) layer).name())) {
                return layer;
            }
        }
        throw new IllegalStateException(
                "rw-agent: no movement layer named '" + name + "'" + EngineNames.PIN);
    }

    /**
     * Returns an entity's owning team number, or -1 when it has no owner.
     *
     * @param entity The entity to read.
     * @return The team number the engine uses in its own AI warnings.
     */
    static int teamOf(Object entity) {
        Object owner = EngineAccess.readField(entity, EngineNames.OWNER);
        if (owner == null) {
            return -1;
        }
        return EngineAccess.readIntField(owner, EngineNames.TEAM_ID);
    }

    /**
     * Returns current and maximum hit points.
     *
     * @param entity The entity to read.
     * @return ``{current, maximum}``.
     */
    static float[] healthOf(Object entity) {
        return new float[] {EngineAccess.readFloat(entity, EngineNames.HP), EngineAccess.readFloat(entity, EngineNames.MAX_HP)};
    }

    /**
     * Returns the current player's team number and credit balance.
     *
     * @param engine The live engine instance.
     * @return ``{team, credits}``, or null when there is no current player.
     */
    static double[] localPlayerState(Object engine) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return null;
        }
        return new double[] {EngineAccess.readIntField(team, EngineNames.TEAM_ID), EngineAccess.readDoubleField(team, EngineNames.CREDITS)};
    }

    /**
     * Lists every entity the current player owns, with the movement fields that
     * decide whether it can be sent anywhere.
     *
     * <p>Written after ordering a Command Center to move and watching nothing
     * happen. "First owned unit" is not a useful selection when the roster is
     * unknown, and the engine will accept an order for an immobile building
     * without complaint -- so the roster has to be legible before selection can
     * be. Reports max-speed and the movement-kind field alongside the class so
     * mobility is read rather than inferred from a package name.
     *
     * @param engine The live engine instance.
     * @return A multi-line report, one line per owned entity.
     */
    static String describeOwned(Object engine) {
        StringBuilder out = new StringBuilder();
        out.append("=== owned EngineAccess.entities ===\n");
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            out.append("no current player\n");
            return out.toString();
        }
        Class<?> treeClass = EngineAccess.pinnedClass(EngineNames.TREE_CLASS);
        int index = 0;
        for (Object entity : EngineAccess.entities()) {
            if (entity == null || treeClass.isInstance(entity)) {
                continue;
            }
            if (EngineAccess.readField(entity, EngineNames.OWNER) != team) {
                continue;
            }
            float[] at = positionOf(entity);
            out.append('[').append(index++).append("] ")
                    .append(entity.getClass().getName())
                    .append(" at (").append(at[0]).append(", ").append(at[1]).append(')')
                    .append(describeMobility(entity))
                    .append('\n');
        }
        if (index == 0) {
            out.append("player owns nothing\n");
        }
        return out.toString();
    }

    /**
     * Renders whatever float fields on an entity look like movement capability.
     *
     * <p>Named fields would be better, but the movement field has not been
     * identified yet and inventing a name would be the guess this method exists
     * to avoid. Reporting every non-zero float on the entity's own class is
     * cheap and lets one run settle it.
     *
     * @param entity The entity to inspect.
     * @return A parenthesised list of non-zero float fields, or empty.
     */
    private static String describeMobility(Object entity) {
        StringBuilder out = new StringBuilder();
        for (Field field : entity.getClass().getDeclaredFields()) {
            if (field.getType() != float.class || java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            field.setAccessible(true);
            float value;
            try {
                value = field.getFloat(entity);
            } catch (IllegalAccessException e) {
                continue;
            }
            if (value != 0.0f) {
                out.append(' ').append(field.getName()).append('=').append(value);
            }
        }
        return out.length() == 0 ? "" : " {" + out.toString().trim() + "}";
    }

    /**
     * Reports whether an entity has finished being built.
     *
     * @param entity The entity to read.
     * @return True once construction is complete.
     */
    static boolean isComplete(Object entity) {
        return predicate(entity, EngineNames.ENTITY_COMPLETE);
    }

    /**
     * Reports whether an entity is airborne at this moment.
     *
     * <p>One of the three the engine's own attackability test branches on, and
     * read per sample because it is state rather than type: a gunship that has
     * landed is a ground target (wiki: policy-combat).
     *
     * @param entity The entity to read.
     * @return True while it is flying.
     */
    static boolean isFlying(Object entity) {
        return predicate(entity, EngineNames.ENTITY_FLYING);
    }

    /**
     * Reports whether an entity is below the surface at this moment.
     *
     * @param entity The entity to read.
     * @return True while it is submerged.
     */
    static boolean isSubmerged(Object entity) {
        return predicate(entity, EngineNames.ENTITY_SUBMERGED);
    }

    /**
     * Reports whether an entity is standing in water at this moment.
     *
     * @param entity The entity to read.
     * @return True while it is touching water.
     */
    static boolean isTouchingWater(Object entity) {
        return predicate(entity, EngineNames.ENTITY_TOUCHING_WATER);
    }

    /**
     * Asks an entity one of its no-argument boolean predicates.
     *
     * <p>Shared by every predicate above rather than written out per accessor.
     * The failure they all need is identical — a pinned name that has moved
     * reports as "did not return a boolean" naming itself — and four copies of
     * it is four places for that message to drift.
     *
     * @param entity The entity to read.
     * @param name Pinned accessor name.
     * @return The entity's own answer.
     * @throws IllegalStateException When the accessor does not return a boolean.
     */
    private static boolean predicate(Object entity, String name) {
        Object answer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(entity.getClass(), name), entity);
        if (!(answer instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + name + "() did not return a boolean" + EngineNames.PIN);
        }
        return ((Boolean) answer).booleanValue();
    }

    /**
     * Returns how many units a building has queued for production.
     *
     * <p>The engine's public queue accessor reports zero for a factory — it is
     * overridden only by other kinds of producer — so the queue itself is read
     * instead. An entity with no queue field makes nothing, and reports zero
     * for that reason rather than by failing.
     *
     * <p>This is what distinguishes an order the engine accepted from one it
     * dropped. A production order produces no roster change until the unit is
     * finished, so without the queue there is nothing to tell the two apart
     * until a timeout expires (wiki: mechanics-build-actions).
     *
     * @param entity The entity to read.
     * @return The number of items queued, or zero when it has no queue.
     */
    static int queuedCountOf(Object entity) {
        java.lang.reflect.Field queueField =
                EngineAccess.fieldIfPresent(entity.getClass(), EngineNames.PRODUCTION_QUEUE);
        // Matched by type as well as name. Obfuscation reuses single letters,
        // and reading an unrelated field of the same name off another unit
        // class is not a hypothetical -- it crashed a live run.
        if (queueField == null
                || !EngineAccess.pinnedClass(EngineNames.QUEUE_CLASS)
                        .isAssignableFrom(queueField.getType())) {
            return 0;
        }
        Object queue;
        try {
            queue = queueField.get(entity);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read " + EngineNames.PRODUCTION_QUEUE + EngineNames.PIN, e);
        }
        if (queue == null) {
            return 0;
        }
        Object items = EngineAccess.readField(queue, EngineNames.QUEUE_ITEMS);
        if (!(items instanceof java.util.Collection)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.QUEUE_ITEMS + " is not a collection"
                            + EngineNames.PIN);
        }
        return ((java.util.Collection<?>) items).size();
    }

    /**
     * Reads an entity's world position.
     *
     * @param entity The entity to read.
     * @return Its x and y, in that order.
     */
    static float[] positionOf(Object entity) {
        return new float[] {EngineAccess.readFloat(entity, EngineNames.POS_X), EngineAccess.readFloat(entity, EngineNames.POS_Y)};
    }

    /**
     * Renders an entity for a log line: its class and position.
     *
     * @param entity The entity to describe.
     * @return A short identifying string.
     */
    static String describe(Object entity) {
        if (entity == null) {
            return "none";
        }
        float[] at = positionOf(entity);
        return entity.getClass().getName() + " at (" + at[0] + ", " + at[1] + ")";
    }

    /**
     * Returns the current player's credits, rounded down to whole currency.
     *
     * <p>The engine holds this as a double and spends it in whole units, so a
     * planner comparing against a unit price wants the floor rather than the
     * raw value: 99.97 credits does not buy a 100-credit structure.
     *
     * @param engine The live engine instance.
     * @return Credits, or 0 when there is no current player.
     */
    static int creditsOf(Object engine) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return 0;
        }
        return (int) Math.floor(EngineAccess.readDoubleField(team, EngineNames.CREDITS));
    }

    /**
     * Returns an entity's engine-assigned identity.
     *
     * <p>Assigned once at construction, guarded by an "ID for GameObject is
     * already set" throw, and used by the engine itself for network identity.
     * That makes it the only stable handle for addressing a unit across
     * frames: roster position renumbers whenever anything is built or dies.
     *
     * @param entity The entity to identify.
     * @return Its identity.
     */
    static long idOf(Object entity) {
        return EngineAccess.readLongField(entity, EngineNames.ENTITY_ID);
    }

    /**
     * Returns an entity's readable type name, e.g. {@code "builder"}.
     *
     * <p>The engine reaches this as {@code entity.r().i()} -- the unit type,
     * then its name. Both hops are obfuscated; the name they yield is not, and
     * it is the same string the type registry accepts when building.
     *
     * @param entity The entity to name.
     * @return Its type name.
     */
    static String typeNameOf(Object entity) {
        Object type = EngineAccess.invoke(EngineAccess.pinnedMethod(entity.getClass(), EngineNames.TYPE_ACCESSOR), entity);
        if (type == null) {
            throw new IllegalStateException("rw-agent: entity has no unit type" + EngineNames.PIN);
        }
        return nameOfType(type);
    }

    /**
     * Returns a unit type's readable name.
     *
     * <p>Split from {@link #typeNameOf} because a type is reached two ways. An
     * entity carries one, and so does a build action that produces it; both
     * name it identically, and both names are the string a plan and a build
     * order use.
     *
     * @param type The unit type.
     * @return Its name.
     */
    static String nameOfType(Object type) {
        Object name = EngineAccess.invoke(EngineAccess.pinnedMethod(type.getClass(), EngineNames.TYPE_NAME_ACCESSOR), type);
        if (!(name instanceof String)) {
            throw new IllegalStateException("rw-agent: unit type name is not a String" + EngineNames.PIN);
        }
        return (String) name;
    }

    /**
     * Reports whether the current player has been defeated.
     *
     * <p>The engine's own verdict, not a count of what is left standing. It
     * fires a notification reading "&lt;player&gt; was defeated" on the same
     * transition, which is what pins the flag (wiki: policy-grading).
     *
     * @param engine The live engine instance.
     * @return True when the current player is out of the match.
     */
    static boolean isDefeated(Object engine) {
        return playerFlag(engine, EngineNames.PLAYER_DEFEATED);
    }

    /**
     * Reports whether the current player has been wiped out.
     *
     * <p>Stronger than defeat: nothing owned is left at all, and no ally holds
     * anything either. Its notification reads "&lt;player&gt; has been wiped
     * out".
     *
     * @param engine The live engine instance.
     * @return True when the current player holds nothing.
     */
    static boolean isWipedOut(Object engine) {
        return playerFlag(engine, EngineNames.PLAYER_WIPED);
    }

    /** Reads one boolean flag off the current player. */
    private static boolean playerFlag(Object engine, String field) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return false;
        }
        Object value = EngineAccess.readField(team, field);
        if (!(value instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: player field " + field + " is not a boolean" + EngineNames.PIN);
        }
        return ((Boolean) value).booleanValue();
    }

    /**
     * One player's scoreboard, as the engine keeps it.
     *
     * <p>Carried per player rather than for the local one alone, because the
     * question worth asking is comparative. "Our army is worth 3,400" says
     * nothing; "ours is 3,400 against a leader on 22,000" is the whole of the
     * match report.
     */
    static final class PlayerStat {

        private final int team;
        private final boolean local;
        private final boolean hostile;
        private final boolean defeated;
        private final boolean wiped;
        private final int income;
        private final int armyValue;
        private final int buildingValue;

        PlayerStat(
                int team,
                boolean local,
                boolean hostile,
                boolean defeated,
                boolean wiped,
                int income,
                int armyValue,
                int buildingValue) {
            this.team = team;
            this.local = local;
            this.hostile = hostile;
            this.defeated = defeated;
            this.wiped = wiped;
            this.income = income;
            this.armyValue = armyValue;
            this.buildingValue = buildingValue;
        }

        int team() {
            return this.team;
        }

        boolean local() {
            return this.local;
        }

        boolean hostile() {
            return this.hostile;
        }

        boolean defeated() {
            return this.defeated;
        }

        boolean wiped() {
            return this.wiped;
        }

        int income() {
            return this.income;
        }

        int armyValue() {
            return this.armyValue;
        }

        int buildingValue() {
            return this.buildingValue;
        }
    }

    /**
     * Returns the scoreboard for every player still holding a slot.
     *
     * <p>Read from the engine's own statistics rather than counted here. It
     * keeps income, army value and building value per player, charts all three,
     * and writes them into its own save file — so these are the figures the game
     * itself would show, and a reimplementation could disagree with them
     * (wiki: perception-visibility).
     *
     * <p>Absent slots are skipped and defeated ones are not: a player who has
     * just been eliminated is exactly who a run report wants to name, and their
     * final army value is the measurement that says whether we killed them or
     * somebody else did.
     *
     * @param engine The live engine instance.
     * @return One entry per occupied slot, in slot order.
     * @throws IllegalStateException When the roster or a statistic cannot be
     *     read, which is a pinned name that has moved.
     */
    static java.util.List<PlayerStat> playerStats(Object engine) {
        Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Object localTeam = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        Object roster;
        int size;
        try {
            roster = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER).get(null);
            size = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER_SIZE).getInt(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read the player roster" + EngineNames.PIN, e);
        }
        if (!(roster instanceof Object[])) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TEAM_ROSTER + " is not an array" + EngineNames.PIN);
        }
        Object[] slots = (Object[]) roster;
        java.util.List<PlayerStat> stats = new java.util.ArrayList<PlayerStat>();
        for (int index = 0; index < size && index < slots.length; index++) {
            Object player = slots[index];
            if (player == null || isAbsent(player)) {
                continue;
            }
            stats.add(
                    new PlayerStat(
                            EngineAccess.readIntField(player, EngineNames.TEAM_ID),
                            player == localTeam,
                            localTeam != null && isHostileBetween(localTeam, player),
                            EngineAccess.readBooleanField(player, EngineNames.PLAYER_DEFEATED),
                            EngineAccess.readBooleanField(player, EngineNames.PLAYER_WIPED),
                            statOf(player, EngineNames.STAT_INCOME),
                            statOf(player, EngineNames.STAT_ARMY_VALUE),
                            statOf(player, EngineNames.STAT_BUILDING_VALUE)));
        }
        return stats;
    }

    /**
     * Counts the occupied slots in the player roster.
     *
     * <p>The lobby's join detector: the roster is a static array on the team
     * class, filled as players connect, so a hosted lobby's "someone joined"
     * is a second non-absent slot — no engine reference and no live match
     * needed. Any failure to read counts as zero rather than throwing,
     * because the poller runs from boot and the roster may simply not exist
     * yet.
     */
    static int rosterCount() {
        Object roster;
        int size;
        try {
            Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
            roster = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER).get(null);
            size = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER_SIZE).getInt(null);
        } catch (IllegalAccessException | RuntimeException e) {
            return 0;
        }
        if (!(roster instanceof Object[])) {
            return 0;
        }
        Object[] slots = (Object[]) roster;
        int count = 0;
        for (int index = 0; index < size && index < slots.length; index++) {
            Object player = slots[index];
            if (player != null && !isAbsent(player)) {
                count++;
            }
        }
        return count;
    }

    /** Reports whether a player slot is empty rather than occupied. */
    private static boolean isAbsent(Object player) {
        Object answer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(player.getClass(), EngineNames.TEAM_ABSENT),
                        player);
        if (!(answer instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TEAM_ABSENT + "() did not return a boolean"
                            + EngineNames.PIN);
        }
        return ((Boolean) answer).booleanValue();
    }

    /**
     * Reads one named statistic for one player.
     *
     * <p>The constant is found by its own {@code name()} rather than by ordinal,
     * so a reordered enum fails to find the name instead of silently returning
     * the neighbouring statistic. {@code name()} is final on {@link Enum} and
     * returns a stored string, so nothing engine-side runs to answer it.
     *
     * @param player The player to measure.
     * @param constant The statistic's own constant name.
     * @return The figure.
     * @throws IllegalStateException When the enum carries no such constant, or
     *     the read does not return an int.
     */
    private static int statOf(Object player, String constant) {
        Class<?> statClass = EngineAccess.pinnedClass(EngineNames.PLAYER_STAT_CLASS);
        Object[] constants = statClass.getEnumConstants();
        if (constants == null) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.PLAYER_STAT_CLASS + " is no longer an enum"
                            + EngineNames.PIN);
        }
        for (Object candidate : constants) {
            if (!constant.equals(((Enum<?>) candidate).name())) {
                continue;
            }
            Object value =
                    EngineAccess.invoke(
                            EngineAccess.pinnedMethod(
                                    statClass,
                                    EngineNames.PLAYER_STAT_READ,
                                    EngineAccess.pinnedClass(EngineNames.TEAM_CLASS)),
                            candidate,
                            player);
            if (!(value instanceof Integer)) {
                throw new IllegalStateException(
                        "rw-agent: statistic " + constant + " did not return an int"
                                + EngineNames.PIN);
            }
            return ((Integer) value).intValue();
        }
        throw new IllegalStateException(
                "rw-agent: " + EngineNames.PLAYER_STAT_CLASS + " carries no constant named "
                        + constant + EngineNames.PIN);
    }

    /**
     * Returns how many players are still in the match.
     *
     * <p>Asked of the engine rather than counted here. It excludes absent,
     * defeated and wiped-out players, prints the same figure as "N players
     * remaining", and calls its own end-of-match hook when it reaches one -- so
     * this is the engine's scoreboard, and a reimplementation could disagree
     * with the thing that actually ends the game.
     *
     * @return The count of players still playing.
     */
    static int playersRemaining() {
        Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Object value =
                EngineAccess.invokeStatic(teamClass, EngineNames.PLAYERS_REMAINING);
        if (!(value instanceof Integer)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.PLAYERS_REMAINING
                            + "() did not return an int" + EngineNames.PIN);
        }
        return ((Integer) value).intValue();
    }

    /**
     * Finds an owned entity by its engine identity.
     *
     * @param engine The live engine instance.
     * @param id The identity to find.
     * @return The entity, or null when the current player owns no entity with
     *     that identity -- which is the normal answer for a unit that has died.
     */
    static Object findOwnedById(Object engine, long id) {
        for (Object entity : ownedUnits(engine)) {
            if (idOf(entity) == id) {
                return entity;
            }
        }
        return null;
    }

    /**
     * Finds a currently visible entity by its engine identity.
     *
     * <p>The counterpart of {@link #findOwnedById} for the other side. An
     * attack order names a target the player does not own, so the owned list
     * cannot resolve it -- but the search still runs over what the player can
     * legitimately see rather than the master entity list, or the bot could
     * order an attack on something it has no way of knowing is there (wiki:
     * multiplayer-portability-invariants).
     *
     * @param engine The live engine instance.
     * @param id The identity to find.
     * @return The entity, or null when nothing visible carries that identity --
     *     the normal answer for a target that has died or slipped back into
     *     fog since the sample was taken.
     */
    static Object findVisibleById(Object engine, long id) {
        for (Object entity : visibleEntities(engine)) {
            if (idOf(entity) == id) {
                return entity;
            }
        }
        return null;
    }
}
