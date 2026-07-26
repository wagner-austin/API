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
     * <p>Trees are EngineBindings.entities too, so the tree subclass is excluded explicitly
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
     * @return The owned EngineBindings.entities, in entity-list order. Empty when the player
     *     owns nothing or there is no current player.
     */
    static java.util.List<Object> ownedUnits(Object engine) {
        java.util.List<Object> owned = new java.util.ArrayList<Object>();
        Object team = EngineBindings.readField(engine, EngineBindings.LOCAL_TEAM);
        if (team == null) {
            return owned;
        }
        Class<?> treeClass = EngineBindings.pinnedClass(EngineBindings.TREE_CLASS);
        Class<?> orderableClass = EngineBindings.pinnedClass(EngineBindings.ORDERABLE_CLASS);
        for (Object entity : EngineBindings.entities()) {
            if (entity == null || treeClass.isInstance(entity) || !orderableClass.isInstance(entity)) {
                continue;
            }
            if (EngineBindings.readField(entity, EngineBindings.OWNER) == team) {
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
     * @return The visible EngineBindings.entities, in entity-list order. Empty when there is
     *     no current player.
     */
    static java.util.List<Object> visibleEntities(Object engine) {
        java.util.List<Object> visible = new java.util.ArrayList<Object>();
        Object team = EngineBindings.readField(engine, EngineBindings.LOCAL_TEAM);
        if (team == null) {
            return visible;
        }
        Class<?> treeClass = EngineBindings.pinnedClass(EngineBindings.TREE_CLASS);
        Class<?> orderableClass = EngineBindings.pinnedClass(EngineBindings.ORDERABLE_CLASS);
        Class<?> entityClass = EngineBindings.pinnedClass(EngineBindings.ENTITY_CLASS);
        Method visibleTo = EngineBindings.pinnedMethod(entityClass, EngineBindings.VISIBLE_TO, EngineBindings.pinnedClass(EngineBindings.TEAM_CLASS));
        for (Object entity : EngineBindings.entities()) {
            if (entity == null || treeClass.isInstance(entity) || !orderableClass.isInstance(entity)) {
                continue;
            }
            Object seen = EngineBindings.invoke(visibleTo, entity, team);
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
        return EngineBindings.readField(entity, EngineBindings.OWNER) == EngineBindings.readField(engine, EngineBindings.LOCAL_TEAM);
    }

    /**
     * Returns an entity's owning team number, or -1 when it has no owner.
     *
     * @param entity The entity to read.
     * @return The team number the engine uses in its own AI warnings.
     */
    static int teamOf(Object entity) {
        Object owner = EngineBindings.readField(entity, EngineBindings.OWNER);
        if (owner == null) {
            return -1;
        }
        return EngineBindings.readIntField(owner, EngineBindings.TEAM_ID);
    }

    /**
     * Returns current and maximum hit points.
     *
     * @param entity The entity to read.
     * @return ``{current, maximum}``.
     */
    static float[] healthOf(Object entity) {
        return new float[] {EngineBindings.readFloat(entity, EngineBindings.HP), EngineBindings.readFloat(entity, EngineBindings.MAX_HP)};
    }

    /**
     * Returns the current player's team number and credit balance.
     *
     * @param engine The live engine instance.
     * @return ``{team, credits}``, or null when there is no current player.
     */
    static double[] localPlayerState(Object engine) {
        Object team = EngineBindings.readField(engine, EngineBindings.LOCAL_TEAM);
        if (team == null) {
            return null;
        }
        return new double[] {EngineBindings.readIntField(team, EngineBindings.TEAM_ID), EngineBindings.readDoubleField(team, EngineBindings.CREDITS)};
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
        out.append("=== owned EngineBindings.entities ===\n");
        Object team = EngineBindings.readField(engine, EngineBindings.LOCAL_TEAM);
        if (team == null) {
            out.append("no current player\n");
            return out.toString();
        }
        Class<?> treeClass = EngineBindings.pinnedClass(EngineBindings.TREE_CLASS);
        int index = 0;
        for (Object entity : EngineBindings.entities()) {
            if (entity == null || treeClass.isInstance(entity)) {
                continue;
            }
            if (EngineBindings.readField(entity, EngineBindings.OWNER) != team) {
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
     * Reads an entity's world position.
     *
     * @param entity The entity to read.
     * @return Its x and y, in that order.
     */
    static float[] positionOf(Object entity) {
        return new float[] {EngineBindings.readFloat(entity, EngineBindings.POS_X), EngineBindings.readFloat(entity, EngineBindings.POS_Y)};
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
        Object team = EngineBindings.readField(engine, EngineBindings.LOCAL_TEAM);
        if (team == null) {
            return 0;
        }
        return (int) Math.floor(EngineBindings.readDoubleField(team, EngineBindings.CREDITS));
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
        return EngineBindings.readLongField(entity, EngineBindings.ENTITY_ID);
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
        Object type = EngineBindings.invoke(EngineBindings.pinnedMethod(entity.getClass(), EngineBindings.TYPE_ACCESSOR), entity);
        if (type == null) {
            throw new IllegalStateException("rw-agent: entity has no unit type" + EngineBindings.PIN);
        }
        Object name = EngineBindings.invoke(EngineBindings.pinnedMethod(type.getClass(), EngineBindings.TYPE_NAME_ACCESSOR), type);
        if (!(name instanceof String)) {
            throw new IllegalStateException("rw-agent: unit type name is not a String" + EngineBindings.PIN);
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
}
