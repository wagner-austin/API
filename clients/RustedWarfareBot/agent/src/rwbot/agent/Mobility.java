package rwbot.agent;

import java.lang.reflect.Field;

/**
 * How an entity moves, and which ground it can reach.
 *
 * <p>Reads only, like {@link Perception}, which is where this was split from
 * when that class reached 943 lines. The distinction is that everything here
 * is about <i>travel</i>: the layer a unit rides on, the connectivity component
 * it stands in, and whether a point on the map can be walked to at all. A
 * unit's identity, health and ownership are the roster's business and stay
 * there.
 *
 * <p>Reachability is the engine's own answer reduced to a comparison: two
 * things can reach each other exactly when their component ids match and both
 * are real. Recomputing it here would be a second pathfinder to keep in step
 * with the first (wiki: mechanics-movement-layers).
 */
final class Mobility {

    private Mobility() {
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
        float[] at = Perception.positionOf(entity);
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
            float[] at = Perception.positionOf(entity);
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
}
