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
        if (orderable != null) {
            checkField(orderable, EngineNames.LAST_DAMAGER, problems);
        }
        Class<?> command = checkClass(EngineNames.COMMAND_CLASS, problems);
        Class<?> controller = checkClass(EngineNames.CONTROLLER_CLASS, problems);
        Class<?> scripts = checkClass(EngineNames.SCRIPTS_CLASS, problems);
        Class<?> type = checkClass(TypeNames.TYPE_CLASS, problems);
        Class<?> registry = checkClass(TypeNames.TYPE_REGISTRY_CLASS, problems);
        checkClass(EngineNames.TREE_CLASS, problems);

        if (entity != null) {
            checkField(entity, EngineNames.ENTITY_ID, problems);
            checkMethod(entity, TypeNames.TYPE_ACCESSOR, problems);
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
            // Hostility and the unit cap. Losing either fails quietly rather
            // than loudly: the first would make every enemy read as harmless
            // and send builders into gun range, the second would drop a gate
            // from the produce diagnostic and take a silent refusal with it.
            checkMethod(team, EngineNames.TEAM_HOSTILE_TO, problems, team);
            checkMethod(team, EngineNames.TEAM_UNIT_COUNT, problems);
            checkMethod(team, EngineNames.TEAM_UNIT_CAP, problems);
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
            checkMethod(type, TypeNames.TYPE_NAME_ACCESSOR, problems);
        }
        if (command != null && type != null) {
            checkMethod(command, "a", problems, float.class, float.class, type, int.class);
        }

        Class<?> map = checkClass(EngineNames.MAP_CLASS, problems);
        Class<?> tile = checkClass(EngineNames.TILE_CLASS, problems);
        Class<?> customType = checkClass(TypeNames.CUSTOM_TYPE_CLASS, problems);
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
            checkMethod(type, TypeNames.TYPE_NEEDS_POOL, problems);
        }
        if (customType != null) {
            checkField(customType, TypeNames.CUSTOM_TYPE_LIST, problems);
        }

        Class<?> action = checkClass(TypeNames.ACTION_CLASS, problems);
        if (entity != null) {
            checkMethod(entity, TypeNames.ACTIONS, problems);
        }
        Class<?> actionKey = checkClass(TypeNames.ACTION_KEY_CLASS, problems);
        Class<?> point = checkClass(TypeNames.POINT_CLASS, problems);
        if (action != null) {
            checkMethod(action, TypeNames.ACTION_MAKES, problems);
            checkMethod(action, TypeNames.ACTION_PLACED_TYPE, problems);
            checkMethod(action, TypeNames.ACTION_MAKES_SOMETHING, problems);
            checkMethod(action, TypeNames.ACTION_KEY, problems);
            checkMethod(action, TypeNames.ACTION_INDEX, problems);
        }
        if (command != null && actionKey != null && point != null && entity != null) {
            // The production verb. Distinct from the placement one above, and
            // checked separately, because a game update could move either.
            checkMethod(command, "a", problems, actionKey, point, entity);
        }
        if (action != null && entity != null) {
            checkMethod(action, TypeNames.ACTION_AVAILABLE, problems, entity);
            checkMethod(action, TypeNames.ACTION_LOCKED, problems, entity);
            checkMethod(action, TypeNames.ACTION_APPLIES, problems, entity, boolean.class);
        }
        if (entity != null) {
            checkMethod(entity, EngineNames.ENTITY_COMPLETE, problems);
        }
        Class<?> queue = checkClass(TypeNames.QUEUE_CLASS, problems);
        if (queue != null) {
            checkField(queue, TypeNames.QUEUE_ITEMS, problems);
        }

        // The research probe. Checked like everything else, but note what a
        // failure here does and does not mean: nothing the bot plays with
        // depends on these, so a break costs a dump rather than a run.
        // Terrain connectivity. Losing this fails quietly in the worst
        // direction: every pool would read as unreachable and the economy
        // would stop, which looks like a planner bug rather than a binding one.
        // Attack range for every registered type, which -printunits covers for
        // barely half. Losing these fails in the dangerous direction: every
        // hostile would read as unarmed and the threat filter would pass
        // everything (wiki: policy-threat).
        if (entity != null && type != null) {
            checkMethod(entity, TypeNames.TYPE_PROTOTYPE, problems, type);
            checkMethod(entity, TypeNames.UNIT_ARMED, problems);
        }
        if (orderable != null) {
            checkMethod(orderable, TypeNames.UNIT_ATTACK_RANGE, problems);
        }

        Class<?> pathing = checkClass(EngineNames.PATHING_CLASS, problems);
        Class<?> movement = checkClass(EngineNames.MOVEMENT_CLASS, problems);
        if (pathing != null && movement != null) {
            checkMethod(
                    pathing, EngineNames.PATH_GROUP_AT, problems,
                    float.class, float.class, movement);
        }
        if (entity != null) {
            checkMethod(entity, EngineNames.ENTITY_MOVEMENT, problems);
        }
        if (movement != null && movement.getEnumConstants() == null) {
            problems.add("not an enum: " + EngineNames.MOVEMENT_CLASS);
        }

        Class<?> engine = checkClass(EngineNames.ENGINE_CLASS, problems);
        if (engine != null) {
            // The wrong-world guard. Losing either name fails quietly in the
            // worst direction: a match whose map a game dir lacks would fall
            // back to the boot sandbox and produce a plausible scorecard from
            // a world nobody asked for (wiki: policy-determinism).
            checkField(engine, EngineNames.MAP_PATH, problems);
            checkField(engine, EngineNames.TESTING_FLAG, problems);
            // The synced-draw seed. Losing it quietly would not break a
            // build: same-seed runs would simply stop reproducing across
            // invocations, forking at the first AI decision -- the exact
            // defect it was pinned to close.
            checkField(engine, EngineNames.SYNC_SEED, problems);
        }

        Class<?> ai = checkClass(EngineNames.AI_CLASS, problems);
        checkClass(EngineNames.ZONE_CLASS, problems);
        if (ai != null) {
            checkField(ai, EngineNames.AI_ZONES, problems);
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
