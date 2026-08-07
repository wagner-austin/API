package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * What the bot can see: the roster, positions, health, ownership and state.
 *
 * <p>Reads only. Nothing here writes to the simulation and nothing here decides
 * anything — selection is the planner's job (wiki:
 * runtime-split-java-agent-python-brain). Dispatch is {@link Orders}; the
 * obfuscated names and the reflection are {@link EngineBindings}.
 *
 * <p>Two neighbours answer the questions this one deliberately does not. Where
 * a unit can travel is {@link Mobility}; what the players are worth is {@link
 * Scoreboard}. Both were split out of here, and the seam in each case is the
 * subject of the question rather than the size of the file.
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
    /**
     * Returns the type name of the unit that last damaged this one, or empty.
     *
     * <p>The read the death ledger rides: the engine keeps {@code y.bt}
     * current as damage lands, so the sampler needs no combat events -- the
     * last value published before a unit vanishes names its killer's type.
     * Deliberately unlike the engine's own {@code lastDamagedBy} reference,
     * a damager that has since died still answers: the engine blanks a dead
     * reference because ITS use is targeting, and a corpse cannot be
     * targeted -- but a killer that traded itself is still the killer, and
     * attribution wants the type, not a live object.
     *
     * @param entity The unit to read.
     * @return The last damager's type name, or empty when the entity carries
     *     no such field (map objects), or nothing has damaged it yet.
     */
    static String lastDamagerTypeOf(Object entity) {
        java.lang.reflect.Field field =
                EngineAccess.fieldIfPresent(entity.getClass(), EngineNames.LAST_DAMAGER);
        if (field == null) {
            return "";
        }
        Object damager;
        try {
            damager = field.get(entity);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read " + EngineNames.LAST_DAMAGER + EngineNames.PIN, e);
        }
        if (damager == null) {
            return "";
        }
        return typeNameOf(damager);
    }

    static float[] healthOf(Object entity) {
        return new float[] {EngineAccess.readFloat(entity, EngineNames.HP), EngineAccess.readFloat(entity, EngineNames.MAX_HP)};
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
                EngineAccess.fieldIfPresent(entity.getClass(), TypeNames.PRODUCTION_QUEUE);
        // Matched by type as well as name. Obfuscation reuses single letters,
        // and reading an unrelated field of the same name off another unit
        // class is not a hypothetical -- it crashed a live run.
        if (queueField == null
                || !EngineAccess.pinnedClass(TypeNames.QUEUE_CLASS)
                        .isAssignableFrom(queueField.getType())) {
            return 0;
        }
        Object queue;
        try {
            queue = queueField.get(entity);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read " + TypeNames.PRODUCTION_QUEUE + EngineNames.PIN, e);
        }
        if (queue == null) {
            return 0;
        }
        Object items = EngineAccess.readField(queue, TypeNames.QUEUE_ITEMS);
        if (!(items instanceof java.util.Collection)) {
            throw new IllegalStateException(
                    "rw-agent: " + TypeNames.QUEUE_ITEMS + " is not a collection"
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
        Object type = EngineAccess.invoke(EngineAccess.pinnedMethod(entity.getClass(), TypeNames.TYPE_ACCESSOR), entity);
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
        Object name = EngineAccess.invoke(EngineAccess.pinnedMethod(type.getClass(), TypeNames.TYPE_NAME_ACCESSOR), type);
        if (!(name instanceof String)) {
            throw new IllegalStateException("rw-agent: unit type name is not a String" + EngineNames.PIN);
        }
        return (String) name;
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
