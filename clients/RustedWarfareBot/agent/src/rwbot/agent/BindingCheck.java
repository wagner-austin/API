package rwbot.agent;

/**
 * Resolves every pinned name against the jar, with no game running.
 *
 * <p>The drift guard. Run by {@code make check} through the agent self-checks,
 * so a game update that moves an obfuscated name fails at the gate instead of
 * during a live run, where it would surface as an engine crash log — the
 * hardest place to read a build problem.
 *
 * <p>It reports every unresolved name rather than the first. After an update it
 * is far more useful to see the whole broken surface at once than to fix one
 * name, re-run, and discover the next.
 *
 * <p>One binding here is not like the others. Losing the visibility method
 * would not break a build: it would silently make the bot omniscient, because
 * the fog test would stop being applied. That is worse than a crash, since
 * nothing would look wrong, which is why it is checked alongside the fields
 * rather than trusted to fail loudly on its own.
 */
final class BindingCheck {

    private BindingCheck() {
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

        Class<?> entity = checkClass(EngineNames.ENTITY_CLASS, problems);
        Class<?> team = checkClass(EngineNames.TEAM_CLASS, problems);
        Class<?> orderable = checkClass(EngineNames.ORDERABLE_CLASS, problems);
        Class<?> command = checkClass(EngineNames.COMMAND_CLASS, problems);
        Class<?> controller = checkClass(EngineNames.CONTROLLER_CLASS, problems);
        Class<?> scripts = checkClass(EngineNames.SCRIPTS_CLASS, problems);
        Class<?> type = checkClass(EngineNames.TYPE_CLASS, problems);
        Class<?> registry = checkClass(EngineNames.TYPE_REGISTRY_CLASS, problems);
        checkClass(EngineNames.TREE_CLASS, problems);

        if (entity != null) {
            checkField(entity, EngineNames.ENTITY_ID, problems);
            checkMethod(entity, EngineNames.TYPE_ACCESSOR, problems);
            checkField(entity, EngineNames.ENTITY_LIST, problems);
            checkField(entity, EngineNames.OWNER, problems);
            checkField(entity, EngineNames.POS_X, problems);
            checkField(entity, EngineNames.POS_Y, problems);
            checkField(entity, EngineNames.HP, problems);
            checkField(entity, EngineNames.MAX_HP, problems);
        }
        if (entity != null && team != null) {
            // The fog test. Losing this name silently would not break a build --
            // it would make the bot omniscient, which is worse than a crash
            // because nothing would look wrong.
            checkMethod(entity, EngineNames.VISIBLE_TO, problems, team);
        }
        if (team != null) {
            checkField(team, EngineNames.TEAM_ID, problems);
        }
        if (controller != null && team != null) {
            checkMethod(controller, "a", problems, team);
        }
        if (team != null) {
            checkField(team, EngineNames.CREDITS, problems);
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
            checkMethod(type, EngineNames.TYPE_NAME_ACCESSOR, problems);
        }
        if (command != null && type != null) {
            checkMethod(command, "a", problems, float.class, float.class, type, int.class);
        }

        Class<?> map = checkClass(EngineNames.MAP_CLASS, problems);
        Class<?> tile = checkClass(EngineNames.TILE_CLASS, problems);
        Class<?> customType = checkClass(EngineNames.CUSTOM_TYPE_CLASS, problems);
        if (map != null) {
            checkField(map, EngineNames.MAP_TILES_X, problems);
            checkField(map, EngineNames.MAP_TILES_Y, problems);
            checkField(map, EngineNames.TILE_WIDTH, problems);
            checkField(map, EngineNames.TILE_HEIGHT, problems);
            checkField(map, EngineNames.TILE_CENTRE_X, problems);
            checkField(map, EngineNames.TILE_CENTRE_Y, problems);
            checkMethod(map, EngineNames.TILE_AT, problems, int.class, int.class);
        }
        if (map != null && team != null) {
            // The tile fog test. Losing this name has the same shape of
            // consequence as losing the entity one: not a crash, but a bot that
            // reads resource pools through fog it never lifted.
            checkMethod(map, EngineNames.TILE_VISIBLE_TO, problems, int.class, int.class, team);
        }
        if (tile != null) {
            checkField(tile, EngineNames.TILE_IS_POOL, problems);
        }
        if (type != null) {
            checkMethod(type, EngineNames.TYPE_NEEDS_POOL, problems);
        }
        if (customType != null) {
            checkField(customType, EngineNames.CUSTOM_TYPE_LIST, problems);
        }

        Class<?> action = checkClass(EngineNames.ACTION_CLASS, problems);
        if (entity != null) {
            checkMethod(entity, EngineNames.ACTIONS, problems);
        }
        Class<?> actionKey = checkClass(EngineNames.ACTION_KEY_CLASS, problems);
        Class<?> point = checkClass(EngineNames.POINT_CLASS, problems);
        if (action != null) {
            checkMethod(action, EngineNames.ACTION_MAKES, problems);
            checkMethod(action, EngineNames.ACTION_PLACED_TYPE, problems);
            checkMethod(action, EngineNames.ACTION_MAKES_SOMETHING, problems);
            checkMethod(action, EngineNames.ACTION_KEY, problems);
            checkMethod(action, EngineNames.ACTION_INDEX, problems);
        }
        if (command != null && actionKey != null && point != null && entity != null) {
            // The production verb. Distinct from the placement one above, and
            // checked separately, because a game update could move either.
            checkMethod(command, "a", problems, actionKey, point, entity);
        }
        if (action != null && entity != null) {
            checkMethod(action, EngineNames.ACTION_AVAILABLE, problems, entity);
            checkMethod(action, EngineNames.ACTION_LOCKED, problems, entity);
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
