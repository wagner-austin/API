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
    /**
     * Orders a building to produce a unit.
     *
     * <p>A different verb from {@link #buildAt}, and it has to be. Placing a
     * structure sends a build waypoint carrying a type, which the engine
     * resolves by scanning the subject's actions for one that <em>places</em>
     * that type. A factory's actions place nothing — the unit rolls out of the
     * building — so they are invisible to that lookup, and ordering a factory
     * that way is refused with "can not queue build". Producing a unit instead
     * sends the action's own interned key, which is what the game's own
     * interface sends when the button is pressed (wiki: mechanics-build-actions).
     *
     * <p>The action is found on the unit and its key read off it, rather than
     * composed from the type name. The engine builds those keys by a private
     * convention, and reproducing that convention here would be a guess that
     * happens to work.
     *
     * <p><b>A closed gate stops the order here rather than at the engine.</b>
     * The engine's enqueue path drops a command it will not run and says
     * nothing, so dispatching into a closed gate produces no unit, no error and
     * no way to tell the two apart. Refusing here is not a second opinion about
     * the engine's rules; it is the same reads the engine is about to make,
     * made where the answer can be logged.
     *
     * <p>A closed gate is a refusal rather than a throw. Being at the unit cap
     * or short of credits is an ordinary state of a game in progress, not a
     * defect, and it can clear on its own. A missing action cannot, which is
     * why that one still throws.
     *
     * @param engine The live engine instance.
     * @param producer The building to order.
     * @param typeName The unit type to produce.
     * @throws IllegalStateException When the producer has no action making that
     *     type, naming what it can make instead.
     */
    /**
     * Orders one unit to attack another.
     *
     * <p>The same command path as a move, in a different target mode. A move
     * sets a waypoint to a point; this sets one to a unit, and the engine
     * follows the target as it moves rather than sending the attacker to where
     * it used to be. That distinction is the reason this is a verb of its own
     * rather than a move at the target's current position.
     *
     * <p>The waypoint mode is the engine's own {@code av.b}. Six setters on the
     * command take a unit and each selects a different mode, so which one
     * attacks was not obvious from their shapes; the engine's waypoint renderer
     * draws this one red and the build mode blue, which is what singled it out,
     * and a live run confirmed it by taking a target's hit points down (wiki:
     * policy-combat).
     *
     * <p>Must be called on the game thread; see {@link #onGameThread}.
     *
     * @param engine The live engine instance.
     * @param attacker The unit to order. Must be owned by the current player.
     * @param target The unit to attack.
     * @throws IllegalStateException When any pinned name is absent or the
     *     command cannot be constructed.
     */
    static void attack(Object engine, Object attacker, Object target) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            throw new IllegalStateException("rw-agent: engine has no current player to order for");
        }
        Object controller = EngineAccess.readField(engine, EngineNames.CONTROLLER);
        if (controller == null) {
            throw new IllegalStateException("rw-agent: engine has no CommandController yet");
        }

        Method create =
                EngineAccess.pinnedMethod(
                        controller.getClass(), "a", EngineAccess.pinnedClass(EngineNames.TEAM_CLASS));
        Object command = EngineAccess.invoke(create, controller, team);
        if (command == null) {
            throw new IllegalStateException("rw-agent: CommandController returned no command");
        }

        Method addUnit =
                EngineAccess.pinnedMethod(
                        command.getClass(), "a", EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS));
        EngineAccess.invoke(addUnit, command, attacker);

        Method setTarget =
                EngineAccess.pinnedMethod(
                        command.getClass(), "a", EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS));
        EngineAccess.invoke(setTarget, command, target);
    }

    static void produce(Object engine, Object producer, String typeName) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            throw new IllegalStateException("rw-agent: engine has no current player to produce for");
        }
        Object controller = EngineAccess.readField(engine, EngineNames.CONTROLLER);
        if (controller == null) {
            throw new IllegalStateException("rw-agent: engine has no CommandController yet");
        }

        Object action = BuildOptions.actionMaking(producer, typeName);
        if (action == null) {
            throw new IllegalStateException(
                    "rw-agent: " + Perception.typeNameOf(producer) + " has no action making '"
                            + typeName + "'; it can make "
                            + BuildOptions.describeMakeable(producer) + EngineNames.PIN);
        }

        Method create = EngineAccess.pinnedMethod(controller.getClass(), "a", EngineAccess.pinnedClass(EngineNames.TEAM_CLASS));
        Object command = EngineAccess.invoke(create, controller, team);
        if (command == null) {
            throw new IllegalStateException("rw-agent: CommandController returned no command");
        }

        Method addUnit = EngineAccess.pinnedMethod(command.getClass(), "a", EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS));
        EngineAccess.invoke(addUnit, command, producer);

        Object key =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(action.getClass(), EngineNames.ACTION_KEY),
                        action);
        BuildOptions.Gates gates = BuildOptions.gatesOf(action, producer);
        // The executor drops an action command whose key is null or the
        // engine's "no action" sentinel, and drops it silently -- no log, no
        // effect, indistinguishable from a command that ran and did nothing.
        // Reporting the key is what makes that failure legible.
        Log.info(
                "produce: "
                        + typeName
                        + " via action "
                        + describeKey(key)
                        + " on "
                        + action.getClass().getName()
                        + " "
                        + gates);
        String closed = gates.closed();
        if (closed != null) {
            Log.error(
                    "produce: refusing to dispatch "
                            + typeName
                            + " on "
                            + Perception.typeNameOf(producer)
                            + " -- "
                            + closed);
            return;
        }
        float[] at = Perception.positionOf(producer);
        Method setAction =
                EngineAccess.pinnedMethod(
                        command.getClass(),
                        "a",
                        EngineAccess.pinnedClass(EngineNames.ACTION_KEY_CLASS),
                        EngineAccess.pinnedClass(EngineNames.POINT_CLASS),
                        EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS));
        EngineAccess.invoke(setAction, command, key, newPoint(at[0], at[1]), null);
    }

    /**
     * Renders an action key for a log line.
     *
     * @param key The interned key, or null.
     * @return Its readable name, or a note that there is none.
     */
    private static String describeKey(Object key) {
        if (key == null) {
            return "<null>";
        }
        Object name =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(key.getClass(), EngineNames.ACTION_KEY_NAME),
                        key);
        return name instanceof String ? "'" + name + "'" : "<unnamed>";
    }

    /**
     * Builds the point an action command carries.
     *
     * <p>The producer's own position. The engine's interface sends wherever the
     * player clicked, and for a production action nothing consumes it; sending
     * the building's own position keeps the command well-formed without
     * inventing a destination the planner did not choose.
     *
     * @param x World x.
     * @param y World y.
     * @return A point the command will accept.
     */
    private static Object newPoint(float x, float y) {
        Class<?> pointClass = EngineAccess.pinnedClass(EngineNames.POINT_CLASS);
        try {
            return pointClass
                    .getConstructor(float.class, float.class)
                    .newInstance(Float.valueOf(x), Float.valueOf(y));
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot construct " + EngineNames.POINT_CLASS + EngineNames.PIN, e);
        }
    }

    static Object resolveType(String typeName) {
        Method lookup = EngineAccess.pinnedMethod(EngineAccess.pinnedClass(EngineNames.TYPE_REGISTRY_CLASS), "a", String.class);
        return EngineAccess.invoke(lookup, null, typeName);
    }
}
