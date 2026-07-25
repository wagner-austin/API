package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * Issues real orders through the engine's own command queue.
 *
 * <p>This is the dispatch half of the agent, and it is deliberately the only
 * place that writes to the simulation. It chooses nothing: which unit and which
 * destination arrive as arguments. Selection is the planner's job (wiki:
 * runtime-split-java-agent-python-brain).
 *
 * <p><b>Why a command and not a field write.</b> Setting a unit's position
 * directly would work in single player and desync every peer in multiplayer,
 * because no other client saw a command that would produce that state. The
 * engine's own AI takes the same route this does -- {@code cf.a(team)}, then
 * add units, then set a target -- so a bot using it is issuing the same class
 * of input a player does (wiki: multiplayer-portability-invariants).
 *
 * <p><b>Why the script queue.</b> Commands are enqueued into a plain
 * {@code ArrayList} and drained by the tick, so writing from a probe thread
 * would race the simulation. {@code ScriptEngine.addRunnableToQueue} appends
 * under a lock and runs the runnable from {@code ScriptEngine.update}, on the
 * thread that marks itself as the main script thread. It is the engine's own
 * answer to "run this on the game thread", so the agent uses it rather than
 * inventing a second one.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> Every name
 * below is obfuscated and moves between releases. {@link #verifyBindings()}
 * checks all of them against the jar and is run by {@code make check}, so a
 * game update fails at the gate rather than in a live run.
 */
final class Orders {

    private static final String ENTITY_CLASS = "com.corrodinggames.rts.game.units.am";
    private static final String TREE_CLASS = "com.corrodinggames.rts.game.units.al";
    private static final String ORDERABLE_CLASS = "com.corrodinggames.rts.game.units.y";
    private static final String TEAM_CLASS = "com.corrodinggames.rts.game.n";
    private static final String COMMAND_CLASS = "com.corrodinggames.rts.gameFramework.e";
    private static final String CONTROLLER_CLASS = "com.corrodinggames.rts.gameFramework.c";
    private static final String SCRIPTS_CLASS = "com.corrodinggames.librocket.scripts.ScriptEngine";
    /** The unit-type interface a placement command carries. */
    private static final String TYPE_CLASS = "com.corrodinggames.rts.game.units.as";
    /** Holds the by-name unit-type lookup. */
    private static final String TYPE_REGISTRY_CLASS = "com.corrodinggames.rts.game.units.ar";

    /** Static master entity list on the entity base class; holds units and trees alike. */
    private static final String ENTITY_LIST = "bE";
    /** Owning player on an entity. */
    private static final String OWNER = "bX";
    /** World position on an entity. */
    private static final String POS_X = "eo";
    private static final String POS_Y = "ep";
    /** The engine's current player. */
    private static final String LOCAL_TEAM = "bs";
    /** The engine's CommandController instance. */
    private static final String CONTROLLER = "cf";

    /**
     * Build-action selector meaning "any action that builds this type".
     *
     * <p>The engine's match is {@code selector == -1 || selector == action.t()},
     * so -1 is the only value that does not require knowing a builder's internal
     * action ordering.
     */
    static final int ANY_BUILD_ACTION = -1;

    private Orders() {
    }

    /**
     * Runs a task on the engine's game thread.
     *
     * @param task Work to run. Executed from {@code ScriptEngine.update}.
     * @throws IllegalStateException When the script engine is unreachable, which
     *     means the pinned names moved.
     */
    static void onGameThread(Runnable task) {
        Class<?> scripts = pinnedClass(SCRIPTS_CLASS);
        Object instance = invokeStatic(scripts, "getInstance");
        if (instance == null) {
            throw new IllegalStateException(
                    "rw-agent: ScriptEngine.getInstance() returned null; the engine has not"
                            + " finished starting");
        }
        Method enqueue = pinnedMethod(scripts, "addRunnableToQueue", Runnable.class);
        try {
            enqueue.invoke(instance, task);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("rw-agent: could not queue work on the game thread", e);
        }
    }

    /**
     * Lists every order-taking entity the engine's current player owns.
     *
     * <p>Trees are entities too, so the tree subclass is excluded explicitly
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
     * @return The owned entities, in entity-list order. Empty when the player
     *     owns nothing or there is no current player.
     */
    static java.util.List<Object> ownedUnits(Object engine) {
        java.util.List<Object> owned = new java.util.ArrayList<Object>();
        Object team = readField(engine, LOCAL_TEAM);
        if (team == null) {
            return owned;
        }
        Class<?> treeClass = pinnedClass(TREE_CLASS);
        Class<?> orderableClass = pinnedClass(ORDERABLE_CLASS);
        for (Object entity : entities()) {
            if (entity == null || treeClass.isInstance(entity) || !orderableClass.isInstance(entity)) {
                continue;
            }
            if (readField(entity, OWNER) == team) {
                owned.add(entity);
            }
        }
        return owned;
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
        out.append("=== owned entities ===\n");
        Object team = readField(engine, LOCAL_TEAM);
        if (team == null) {
            out.append("no current player\n");
            return out.toString();
        }
        Class<?> treeClass = pinnedClass(TREE_CLASS);
        int index = 0;
        for (Object entity : entities()) {
            if (entity == null || treeClass.isInstance(entity)) {
                continue;
            }
            if (readField(entity, OWNER) != team) {
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
     * Orders one unit to move to a world position.
     *
     * <p>Must be called on the game thread; see {@link #onGameThread}.
     *
     * @param engine The live engine instance.
     * @param unit The unit to order. Must be owned by the current player.
     * @param x Destination world x.
     * @param y Destination world y.
     * @throws IllegalStateException When any pinned name is absent or the
     *     command cannot be constructed.
     */
    static void moveTo(Object engine, Object unit, float x, float y) {
        Object team = readField(engine, LOCAL_TEAM);
        if (team == null) {
            throw new IllegalStateException("rw-agent: engine has no current player to order for");
        }
        Object controller = readField(engine, CONTROLLER);
        if (controller == null) {
            throw new IllegalStateException("rw-agent: engine has no CommandController yet");
        }

        Method create = pinnedMethod(controller.getClass(), "a", pinnedClass(TEAM_CLASS));
        Object command = invoke(create, controller, team);
        if (command == null) {
            throw new IllegalStateException("rw-agent: CommandController returned no command");
        }

        Method addUnit = pinnedMethod(command.getClass(), "a", pinnedClass(ORDERABLE_CLASS));
        invoke(addUnit, command, unit);

        Method setPoint = pinnedMethod(command.getClass(), "a", float.class, float.class);
        invoke(setPoint, command, Float.valueOf(x), Float.valueOf(y));
    }

    /**
     * Orders one builder to place a building of a named type at a world position.
     *
     * <p>Building is not a special action. The special-action vocabulary is
     * unit abilities -- reclaim, repair, patrol, launchNuke, upgradeT2 -- and
     * carries nothing for construction. Placement instead rides the same
     * waypoint slot a move order uses, in a different target mode: the setter
     * takes a position, a unit type and a build-action selector, and the
     * interpreter routes on the target kind.
     *
     * <p>The selector is passed as {@link #ANY_BUILD_ACTION}. A builder holds a
     * list of build actions, and the engine matches one by type <em>and</em> by
     * selector unless the selector is -1, which means "any action that builds
     * this type". Passing 0 asks for the action whose own index is 0 and
     * silently matches nothing when it is not; the order is then dropped by
     * waypoint validation with no visible effect.
     *
     * <p>Must be called on the game thread; see {@link #onGameThread}.
     *
     * @param engine The live engine instance.
     * @param builder The unit that will construct it.
     * @param typeName Unit-type name as it appears in the type registry, e.g.
     *     {@code "landFactory"}.
     * @param x Placement world x.
     * @param y Placement world y.
     * @throws IllegalStateException When the type name is unknown, or a pinned
     *     name is absent.
     */
    static void buildAt(Object engine, Object builder, String typeName, float x, float y) {
        Object team = readField(engine, LOCAL_TEAM);
        if (team == null) {
            throw new IllegalStateException("rw-agent: engine has no current player to build for");
        }
        Object controller = readField(engine, CONTROLLER);
        if (controller == null) {
            throw new IllegalStateException("rw-agent: engine has no CommandController yet");
        }

        Object type = resolveType(typeName);
        if (type == null) {
            throw new IllegalStateException(
                    "rw-agent: no unit type named '" + typeName + "' in the registry" + PIN);
        }

        Method create = pinnedMethod(controller.getClass(), "a", pinnedClass(TEAM_CLASS));
        Object command = invoke(create, controller, team);
        if (command == null) {
            throw new IllegalStateException("rw-agent: CommandController returned no command");
        }

        Method addUnit = pinnedMethod(command.getClass(), "a", pinnedClass(ORDERABLE_CLASS));
        invoke(addUnit, command, builder);

        Method place =
                pinnedMethod(
                        command.getClass(),
                        "a",
                        float.class,
                        float.class,
                        pinnedClass(TYPE_CLASS),
                        int.class);
        invoke(
                place,
                command,
                Float.valueOf(x),
                Float.valueOf(y),
                type,
                Integer.valueOf(ANY_BUILD_ACTION));
    }

    /**
     * Resolves a unit type by name through the engine's own registry.
     *
     * <p>The registry lookup tries mod-defined types, then a built-in enum, then
     * mod aliases. The built-in enum arm can never match: its constants are
     * obfuscated to single letters and it compares against
     * {@code Enum.name()}. Every name that resolves therefore resolves through
     * the {@code .ini}-defined registry, which is also where the built-in units
     * live.
     *
     * @param typeName Registry name, e.g. {@code "extractorT1"}.
     * @return The unit type, or null when no type carries that name.
     */
    static Object resolveType(String typeName) {
        Method lookup = pinnedMethod(pinnedClass(TYPE_REGISTRY_CLASS), "a", String.class);
        return invoke(lookup, null, typeName);
    }

    /**
     * Reads an entity's world position.
     *
     * @param entity The entity to read.
     * @return Its x and y, in that order.
     */
    static float[] positionOf(Object entity) {
        return new float[] {readFloat(entity, POS_X), readFloat(entity, POS_Y)};
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
            checkField(entity, ENTITY_LIST, problems);
            checkField(entity, OWNER, problems);
            checkField(entity, POS_X, problems);
            checkField(entity, POS_Y, problems);
        }
        if (controller != null && team != null) {
            checkMethod(controller, "a", problems, team);
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
        if (command != null && type != null) {
            checkMethod(command, "a", problems, float.class, float.class, type, int.class);
        }
        return problems;
    }

    // -----------------------------------------------------------------
    // Reflection helpers. Each failure names the pinned build, because the
    // overwhelmingly likely cause is a game update rather than a code bug.
    // -----------------------------------------------------------------

    private static final String PIN = " -- pinned build is 1.15 (code 176, build #28)";

    private static Iterable<?> entities() {
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

    private static Class<?> pinnedClass(String binaryName) {
        try {
            return Class.forName(binaryName, false, Orders.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("rw-agent: class " + binaryName + " not found" + PIN, e);
        }
    }

    private static Field pinnedField(Class<?> owner, String name) {
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

    private static Method pinnedMethod(Class<?> owner, String name, Class<?>... parameters) {
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

    private static Object readField(Object target, String name) {
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

    private static float readFloat(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getFloat(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read float " + name + PIN, e);
        }
    }

    private static Object invoke(Method method, Object target, Object... arguments) {
        try {
            return method.invoke(target, arguments);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("rw-agent: call to " + method.getName() + " failed", e);
        }
    }

    private static Object invokeStatic(Class<?> owner, String name) {
        return invoke(pinnedMethod(owner, name), null);
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
