package rwbot.agent;

import java.lang.reflect.Method;

/**
 * Issues real orders through the engine's own command queue.
 *
 * <p>The dispatch half of the agent, and deliberately the only place that
 * writes to the simulation. It chooses nothing: which unit and which
 * destination arrive as arguments. Selection is the planner's job (wiki:
 * runtime-split-java-agent-python-brain). Reading is {@link Perception}; the
 * obfuscated names are {@link EngineBindings}.
 *
 * <p><b>Why a command and not a field write.</b> Setting a unit's position
 * directly would work in single player and desync every peer in multiplayer,
 * because no other client saw a command that would produce that state. The
 * engine's own AI takes the same route this does — {@code cf.a(team)}, then add
 * units, then set a target — so a bot using it is issuing the same class of
 * input a player does (wiki: multiplayer-portability-invariants).
 *
 * <p><b>Why the script queue.</b> Commands are enqueued into a plain
 * {@code ArrayList} and drained by the tick, so writing from a probe thread
 * would race the simulation. {@code ScriptEngine.addRunnableToQueue} appends
 * under a lock and runs the runnable from {@code ScriptEngine.update}, on the
 * thread that marks itself as the main script thread. It is the engine's own
 * answer to "run this on the game thread", so the agent uses it rather than
 * inventing a second one.
 */
final class Orders {

    private Orders() {
    }

    /**
     * Build-action selector meaning "any action that builds this type".
     *
     * <p>The engine's match is {@code selector == -1 || selector == action.t()},
     * so -1 is the only value that does not require knowing a builder's internal
     * action ordering.
     */
    static final int ANY_BUILD_ACTION = -1;

    /**
     * Runs a task on the engine's game thread.
     *
     * @param task Work to run. Executed from {@code ScriptEngine.update}.
     * @throws IllegalStateException When the script engine is unreachable, which
     *     means the pinned names moved.
     */
    static void onGameThread(Runnable task) {
        Class<?> scripts = EngineAccess.pinnedClass(EngineNames.SCRIPTS_CLASS);
        Object instance = EngineAccess.invokeStatic(scripts, "getInstance");
        if (instance == null) {
            throw new IllegalStateException(
                    "rw-agent: ScriptEngine.getInstance() returned null; the engine has not"
                            + " finished starting");
        }
        Method enqueue = EngineAccess.pinnedMethod(scripts, "addRunnableToQueue", Runnable.class);
        try {
            enqueue.invoke(instance, task);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("rw-agent: could not queue work on the game thread", e);
        }
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
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            throw new IllegalStateException("rw-agent: engine has no current player to order for");
        }
        Object controller = EngineAccess.readField(engine, EngineNames.CONTROLLER);
        if (controller == null) {
            throw new IllegalStateException("rw-agent: engine has no CommandController yet");
        }

        Method create = EngineAccess.pinnedMethod(controller.getClass(), "a", EngineAccess.pinnedClass(EngineNames.TEAM_CLASS));
        Object command = EngineAccess.invoke(create, controller, team);
        if (command == null) {
            throw new IllegalStateException("rw-agent: CommandController returned no command");
        }

        Method addUnit = EngineAccess.pinnedMethod(command.getClass(), "a", EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS));
        EngineAccess.invoke(addUnit, command, unit);

        Method setPoint = EngineAccess.pinnedMethod(command.getClass(), "a", float.class, float.class);
        EngineAccess.invoke(setPoint, command, Float.valueOf(x), Float.valueOf(y));
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
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            throw new IllegalStateException("rw-agent: engine has no current player to build for");
        }
        Object controller = EngineAccess.readField(engine, EngineNames.CONTROLLER);
        if (controller == null) {
            throw new IllegalStateException("rw-agent: engine has no CommandController yet");
        }

        Object type = resolveType(typeName);
        if (type == null) {
            throw new IllegalStateException(
                    "rw-agent: no unit type named '" + typeName + "' in the registry" + EngineNames.PIN);
        }

        Method create = EngineAccess.pinnedMethod(controller.getClass(), "a", EngineAccess.pinnedClass(EngineNames.TEAM_CLASS));
        Object command = EngineAccess.invoke(create, controller, team);
        if (command == null) {
            throw new IllegalStateException("rw-agent: CommandController returned no command");
        }

        Method addUnit = EngineAccess.pinnedMethod(command.getClass(), "a", EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS));
        EngineAccess.invoke(addUnit, command, builder);

        Method place =
                EngineAccess.pinnedMethod(
                        command.getClass(),
                        "a",
                        float.class,
                        float.class,
                        EngineAccess.pinnedClass(EngineNames.TYPE_CLASS),
                        int.class);
        EngineAccess.invoke(
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
        Method lookup = EngineAccess.pinnedMethod(EngineAccess.pinnedClass(EngineNames.TYPE_REGISTRY_CLASS), "a", String.class);
        return EngineAccess.invoke(lookup, null, typeName);
    }
}
