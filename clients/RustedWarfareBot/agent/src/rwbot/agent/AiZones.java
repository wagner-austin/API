package rwbot.agent;

import java.lang.reflect.Field;

/**
 * Dumps the shipped AI's zone objects, for research and for nothing else.
 *
 * <p><b>This is an instrument, not perception, and the distinction is the whole
 * reason the class is shaped the way it is.</b> A zone is the AI's own plan:
 * where it intends to expand, how large a group must be before it attacks, how
 * long it has been staging. A human player cannot see any of that — they infer
 * it from units they can observe — so a planner reading it would be cheating in
 * a way no amount of care could launder (wiki: engine-ai-zones).
 *
 * <p>Two things keep that from happening by accident rather than by discipline.
 *
 * <p><b>It never reaches the wire.</b> The output goes to the agent log. The
 * planner reads the NDJSON stream and nothing else, so as long as no record
 * kind carries this, the Python side <em>cannot</em> consume it — not "does
 * not", cannot. Anyone wiring it in would have to add a record kind and a
 * decoder, which is a deliberate act rather than a slip.
 *
 * <p><b>It is opt-in and off by default.</b> A run that did not ask for it does
 * not produce it, so an archived capture cannot quietly contain it.
 *
 * <p>There is also a practical reason not to build on this. Zones exist only on
 * AI players. The local human player is constructed as a different subclass
 * entirely, so against a human opponent there is nothing here to read — a
 * policy resting on this would work against the shipped AI and silently do
 * nothing against a person, which is exactly the failure the portability
 * invariant exists to prevent (wiki: multiplayer-portability-invariants).
 *
 * <p><b>Fields are rendered generically rather than by pinned name.</b> That is
 * deliberate for a probe whose purpose is to confirm what the names mean: a
 * dump that applied this reading of the obfuscated letters could only ever
 * agree with it. Reporting every declared field lets the reading be checked
 * against the numbers instead.
 */
final class AiZones {

    private AiZones() {
    }

    /**
     * Renders every zone belonging to every AI player that owns a unit.
     *
     * <p>Players are reached through the entities they own rather than through
     * the engine's player table, which keeps this probe to one new pinned name
     * beyond the AI and zone classes themselves. An AI owning nothing is
     * therefore not reported, which costs nothing: a player with no units has
     * no strategy worth measuring.
     *
     * <p>Read off the game thread, like the discovery snapshots it shares a
     * schedule with. The zone collection is a concurrent queue, so iterating it
     * from another thread is the engine's own supported use.
     *
     * @param engine The live engine instance.
     * @return A multi-line report, one block per AI player.
     */
    static String describe(Object engine) {
        StringBuilder out = new StringBuilder();
        out.append("=== AI zones (research instrument -- never on the wire) ===\n");

        Object localTeam = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        Class<?> aiClass = EngineAccess.pinnedClass(EngineNames.AI_CLASS);
        Class<?> zoneClass = EngineAccess.pinnedClass(EngineNames.ZONE_CLASS);
        Field zoneList = EngineAccess.pinnedField(aiClass, EngineNames.AI_ZONES);

        java.util.List<Object> players = new java.util.ArrayList<Object>();
        for (Object entity : EngineAccess.entities()) {
            if (entity == null) {
                continue;
            }
            Object owner = EngineAccess.readField(entity, EngineNames.OWNER);
            if (owner == null || !aiClass.isInstance(owner) || players.contains(owner)) {
                continue;
            }
            players.add(owner);
        }
        if (players.isEmpty()) {
            out.append("no AI player owns a unit; nothing to report\n");
            return out.toString();
        }

        for (Object player : players) {
            out.append("--- player team ")
                    .append(EngineAccess.readIntField(player, EngineNames.TEAM_ID))
                    .append(player == localTeam ? " (ours) " : " ")
                    .append("credits=")
                    .append((long) EngineAccess.readDoubleField(player, EngineNames.CREDITS))
                    .append(" ---\n");
            appendZones(out, ObjectView.readQuietly(player, zoneList), zoneClass);
        }
        return out.toString();
    }

    /**
     * Renders one player's zones, or says why it could not.
     *
     * @param out Report being built.
     * @param zones The zone collection read off the player.
     * @param zoneClass The pinned zone base class, for rejecting anything else.
     */
    private static void appendZones(StringBuilder out, Object zones, Class<?> zoneClass) {
        if (zones == null) {
            out.append("  zone list is null\n");
            return;
        }
        java.util.Collection<?> held = ObjectView.containedValues(zones);
        if (held == null) {
            out.append("  ")
                    .append(EngineNames.AI_ZONES)
                    .append(" is not a collection; it is ")
                    .append(zones.getClass().getName())
                    .append(EngineNames.PIN)
                    .append('\n');
            return;
        }
        if (held.isEmpty()) {
            out.append("  no zones\n");
            return;
        }
        for (Object zone : held) {
            if (zone == null) {
                continue;
            }
            // Matched by the pinned base class rather than trusted from the
            // field's declared element type, which is erased. Reading an
            // unrelated object's fields as a zone would produce a plausible
            // table of numbers, and a plausible wrong answer is the failure
            // mode this project keeps paying for.
            if (!zoneClass.isInstance(zone)) {
                out.append("  skipped non-zone ").append(zone.getClass().getName()).append('\n');
                continue;
            }
            appendZone(out, zone);
        }
    }

    /** Renders one zone: its class, then every field it declares. */
    private static void appendZone(StringBuilder out, Object zone) {
        out.append("  ").append(zone.getClass().getSimpleName()).append('\n');
        for (Class<?> type : ObjectView.ownedHierarchy(zone.getClass())) {
            for (Field field : ObjectView.declaredFieldsOf(type)) {
                if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
                    continue;
                }
                out.append("    ").append(ObjectView.describeField(zone, field)).append('\n');
            }
        }
    }
}
