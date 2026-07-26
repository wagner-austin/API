package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * Every obfuscated name the agent depends on, and the reflection that reaches
 * them.
 *
 * <p>Split out of {@code Orders} so that one concern lives in one place: this
 * class knows <em>how to reach</em> the engine, {@link Perception} knows what
 * to read from it, and {@link Orders} knows how to write to it. Nothing here
 * decides anything or interprets what it reads.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> Every name
 * below moves between releases. {@link #verifyBindings()} resolves all of them
 * against the jar with no game running and is run by {@code make check}, so a
 * game update fails at the gate rather than in a live run. Each reflection
 * failure names the pinned build, because the overwhelmingly likely cause is a
 * game update rather than a code bug.
 */
final class EngineBindings {

    static final String ENTITY_CLASS = "com.corrodinggames.rts.game.units.am";

    static final String TREE_CLASS = "com.corrodinggames.rts.game.units.al";

    static final String ORDERABLE_CLASS = "com.corrodinggames.rts.game.units.y";

    static final String TEAM_CLASS = "com.corrodinggames.rts.game.n";

    static final String COMMAND_CLASS = "com.corrodinggames.rts.gameFramework.e";

    static final String CONTROLLER_CLASS = "com.corrodinggames.rts.gameFramework.c";

    static final String SCRIPTS_CLASS = "com.corrodinggames.librocket.scripts.ScriptEngine";

    /** The unit-type interface a placement command carries. */
    static final String TYPE_CLASS = "com.corrodinggames.rts.game.units.as";

    /** Holds the by-name unit-type lookup. Also the built-in unit-type enum. */
    static final String TYPE_REGISTRY_CLASS = "com.corrodinggames.rts.game.units.ar";

    /**
     * An asset-defined unit type, as opposed to a built-in enum constant.
     *
     * <p>Most of what the game ships is defined this way: {@code assets/units/}
     * holds an {@code .ini} per unit and the loader turns each into one of
     * these. Both kinds implement {@link #TYPE_CLASS}, so once enumerated they
     * are read identically.
     */
    static final String CUSTOM_TYPE_CLASS = "com.corrodinggames.rts.game.units.custom.l";

    /** The map: terrain grid, tile layers, and the per-tile fog test. */
    static final String MAP_CLASS = "com.corrodinggames.rts.game.b.b";

    /** One map tile, carrying the flags its tileset declared. */
    static final String TILE_CLASS = "com.corrodinggames.rts.game.b.g";

    /** Static master entity list on the entity base class; holds units and trees alike. */
    static final String ENTITY_LIST = "bE";

    /** Owning player on an entity. */
    static final String OWNER = "bX";

    /** World position on an entity. */
    static final String POS_X = "eo";

    static final String POS_Y = "ep";

    /** The engine's current player. */
    static final String LOCAL_TEAM = "bs";

    /** Team number on a player, as the engine prints it in its own AI warnings. */
    static final String TEAM_ID = "k";

    /** Current and maximum hit points on an entity. */
    static final String HP = "cu";

    static final String MAX_HP = "cv";

    /**
     * The engine's own per-player visibility test, {@code am.d(n)}.
     *
     * <p>Its body fog-tests the entity's cell against the asking player's fog
     * grid and returns false when the cell reads as hidden. Own units short out
     * before the test. Using it is what keeps perception legitimate: the master
     * entity list holds every unit on the map, so enumerating that directly
     * would give the bot perfect information and stop it playing the same game a
     * human plays (wiki: multiplayer-portability-invariants).
     */
    static final String VISIBLE_TO = "d";

    /** The engine's CommandController instance. */
    static final String CONTROLLER = "cf";

    /** Credits held by a player. Sits beside the engine's own note to modders. */
    static final String CREDITS = "o";

    /** Engine-assigned object identity, set once at construction. */
    static final String ENTITY_ID = "eh";

    /** Entity accessor returning its unit type. */
    static final String TYPE_ACCESSOR = "r";

    /** Unit-type accessor returning its readable name. */
    static final String TYPE_NAME_ACCESSOR = "i";

    /**
     * Unit-type predicate: true when the type may only be placed on a resource
     * pool.
     *
     * <p>This is the engine's own placement rule, not a reading of the unit's
     * blurb. The chain is: an {@code .ini} declares
     * {@code placeOnlyOnResPool: true}, the loader stores it, this accessor
     * reports it, and the placement check consults it before anything else
     * (wiki: mechanics-resource-pools).
     */
    static final String TYPE_NEEDS_POOL = "p";

    /** The engine's map instance. */
    static final String MAP = "bL";

    /** Map size in tiles. Positions are world units; these are not. */
    static final String MAP_TILES_X = "C";

    static final String MAP_TILES_Y = "D";

    /** Tile size in world units. 20 on desktop, and read rather than assumed. */
    static final String TILE_WIDTH = "n";

    static final String TILE_HEIGHT = "o";

    /** Offset from a tile's origin to its centre, which is where a building goes. */
    static final String TILE_CENTRE_X = "p";

    static final String TILE_CENTRE_Y = "q";

    /**
     * Map accessor returning the item-layer tile at a tile coordinate.
     *
     * <p>The item layer, not the ground layer: resource pools are items. This
     * is the same accessor the engine's own extractor-placement check calls,
     * which is why it is this one and not the ground-layer sibling.
     */
    static final String TILE_AT = "e";

    /**
     * The engine's per-tile fog test, {@code b.a(int, int, n)}.
     *
     * <p>The tile counterpart of {@link #VISIBLE_TO}, and the same predicate:
     * the entity test converts a position to a tile and applies exactly this
     * comparison against the asking player's fog grid. Using it is what stops
     * the bot reading pools through fog it has not lifted.
     */
    static final String TILE_VISIBLE_TO = "a";

    /** Tile flag set by the tileset property {@code res_pool}. */
    static final String TILE_IS_POOL = "i";

    /** Static list of every asset-defined unit type. */
    static final String CUSTOM_TYPE_LIST = "d";

    static final String PIN = " -- pinned build is 1.15 (code 176, build #28)";

    private EngineBindings() {
    }

    static Iterable<?> entities() {
        Class<?> entityClass = pinnedClass(ENTITY_CLASS);
        Field field = pinnedField(entityClass, ENTITY_LIST);
        Object value;
        try {
            value = field.get(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("rw-agent: cannot read " + ENTITY_LIST + PIN, e);
        }
        if (!(value instanceof Iterable)) {
            throw new IllegalStateException(
                    "rw-agent: " + ENTITY_CLASS + "." + ENTITY_LIST + " is not iterable" + PIN);
        }
        return (Iterable<?>) value;
    }

    static Class<?> pinnedClass(String binaryName) {
        try {
            return Class.forName(binaryName, false, Orders.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("rw-agent: class " + binaryName + " not found" + PIN, e);
        }
    }

    static Field pinnedField(Class<?> owner, String name) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                Field field = type.getDeclaredField(name);
                field.setAccessible(true);
                return field;
            } catch (NoSuchFieldException e) {
                continue;
            }
        }
        throw new IllegalStateException(
                "rw-agent: field " + name + " not found on " + owner.getName() + PIN);
    }

    static Method pinnedMethod(Class<?> owner, String name, Class<?>... parameters) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                Method method = type.getDeclaredMethod(name, parameters);
                method.setAccessible(true);
                return method;
            } catch (NoSuchMethodException e) {
                continue;
            }
        }
        throw new IllegalStateException(
                "rw-agent: method "
                        + name
                        + java.util.Arrays.toString(parameters)
                        + " not found on "
                        + owner.getName()
                        + PIN);
    }

    static Object readField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).get(target);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("rw-agent: cannot read " + name + PIN, e);
        }
    }

    /**
     * Reads an {@code int} field through the same pinned-name machinery as
     * every other engine read, so a moved name fails identically here.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not an int.
     */
    static int readIntField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getInt(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read int " + name + PIN, e);
        }
    }

    /**
     * Reads a {@code long} field through the same pinned-name machinery.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not a long.
     */
    static long readLongField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getLong(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read long " + name + PIN, e);
        }
    }

    /**
     * Reads a {@code double} field through the same pinned-name machinery.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not a double.
     */
    static double readDoubleField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getDouble(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read double " + name + PIN, e);
        }
    }

    /**
     * Reads a {@code boolean} field through the same pinned-name machinery.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not a boolean.
     */
    static boolean readBooleanField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getBoolean(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read boolean " + name + PIN, e);
        }
    }

    static float readFloat(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getFloat(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read float " + name + PIN, e);
        }
    }

    static Object invoke(Method method, Object target, Object... arguments) {
        try {
            return method.invoke(target, arguments);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("rw-agent: call to " + method.getName() + " failed", e);
        }
    }

    static Object invokeStatic(Class<?> owner, String name) {
        return invoke(pinnedMethod(owner, name), null);
    }

    /**
     * Checks every pinned name the order path depends on.
     *
     * <p>Runs without a live game -- the jar alone is enough -- so a game update
     * that moves an obfuscated name fails at {@code make check} rather than
     * during a run. Reports every problem rather than the first, because after
     * an update it is more useful to see the whole surface at once.
     *
     * @return One message per broken binding; empty when all resolve.
     */
    static java.util.List<String> verifyBindings() {
        java.util.List<String> problems = new java.util.ArrayList<String>();

        Class<?> entity = checkClass(ENTITY_CLASS, problems);
        Class<?> team = checkClass(TEAM_CLASS, problems);
        Class<?> orderable = checkClass(ORDERABLE_CLASS, problems);
        Class<?> command = checkClass(COMMAND_CLASS, problems);
        Class<?> controller = checkClass(CONTROLLER_CLASS, problems);
        Class<?> scripts = checkClass(SCRIPTS_CLASS, problems);
        Class<?> type = checkClass(TYPE_CLASS, problems);
        Class<?> registry = checkClass(TYPE_REGISTRY_CLASS, problems);
        checkClass(TREE_CLASS, problems);

        if (entity != null) {
            checkField(entity, ENTITY_ID, problems);
            checkMethod(entity, TYPE_ACCESSOR, problems);
            checkField(entity, ENTITY_LIST, problems);
            checkField(entity, OWNER, problems);
            checkField(entity, POS_X, problems);
            checkField(entity, POS_Y, problems);
            checkField(entity, HP, problems);
            checkField(entity, MAX_HP, problems);
        }
        if (entity != null && team != null) {
            // The fog test. Losing this name silently would not break a build --
            // it would make the bot omniscient, which is worse than a crash
            // because nothing would look wrong.
            checkMethod(entity, VISIBLE_TO, problems, team);
        }
        if (team != null) {
            checkField(team, TEAM_ID, problems);
        }
        if (controller != null && team != null) {
            checkMethod(controller, "a", problems, team);
        }
        if (team != null) {
            checkField(team, CREDITS, problems);
        }
        if (command != null && orderable != null) {
            checkMethod(command, "a", problems, orderable);
            checkMethod(command, "a", problems, float.class, float.class);
        }
        if (scripts != null) {
            checkMethod(scripts, "getInstance", problems);
            checkMethod(scripts, "addRunnableToQueue", problems, Runnable.class);
        }
        if (registry != null) {
            checkMethod(registry, "a", problems, String.class);
        }
        if (type != null) {
            checkMethod(type, TYPE_NAME_ACCESSOR, problems);
        }
        if (command != null && type != null) {
            checkMethod(command, "a", problems, float.class, float.class, type, int.class);
        }

        Class<?> map = checkClass(MAP_CLASS, problems);
        Class<?> tile = checkClass(TILE_CLASS, problems);
        Class<?> customType = checkClass(CUSTOM_TYPE_CLASS, problems);
        if (map != null) {
            checkField(map, MAP_TILES_X, problems);
            checkField(map, MAP_TILES_Y, problems);
            checkField(map, TILE_WIDTH, problems);
            checkField(map, TILE_HEIGHT, problems);
            checkField(map, TILE_CENTRE_X, problems);
            checkField(map, TILE_CENTRE_Y, problems);
            checkMethod(map, TILE_AT, problems, int.class, int.class);
        }
        if (map != null && team != null) {
            // The tile fog test. Losing this name has the same shape of
            // consequence as losing the entity one: not a crash, but a bot that
            // reads resource pools through fog it never lifted.
            checkMethod(map, TILE_VISIBLE_TO, problems, int.class, int.class, team);
        }
        if (tile != null) {
            checkField(tile, TILE_IS_POOL, problems);
        }
        if (type != null) {
            checkMethod(type, TYPE_NEEDS_POOL, problems);
        }
        if (customType != null) {
            checkField(customType, CUSTOM_TYPE_LIST, problems);
        }
        return problems;
    }

    private static Class<?> checkClass(String binaryName, java.util.List<String> problems) {
        try {
            return Class.forName(binaryName, false, Orders.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            problems.add("class missing: " + binaryName);
            return null;
        }
    }

    private static void checkField(Class<?> owner, String name, java.util.List<String> problems) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                type.getDeclaredField(name);
                return;
            } catch (NoSuchFieldException e) {
                continue;
            }
        }
        problems.add("field missing: " + owner.getName() + "." + name);
    }

    private static void checkMethod(
            Class<?> owner, String name, java.util.List<String> problems, Class<?>... parameters) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                type.getDeclaredMethod(name, parameters);
                return;
            } catch (NoSuchMethodException e) {
                continue;
            }
        }
        problems.add(
                "method missing: "
                        + owner.getName()
                        + "."
                        + name
                        + java.util.Arrays.toString(parameters));
    }
}
