package rwbot.agent;

/**
 * Turns one parsed order into the engine calls that carry it out.
 *
 * <p>Split out of {@link CommandChannel} when that class reached 619 lines. The
 * seam is real rather than arithmetic: everything else in the channel is about
 * the <i>connection</i> -- accepting a planner, pacing the simulation against
 * it, dropping stale samples under backpressure -- and holds mutable state to
 * do it. This holds none. A record goes in and orders come out, which is why it
 * can be static and why it reads without the socket beside it.
 *
 * <p>Distinct from {@link Orders}, which is the layer below: {@code Orders}
 * knows how to make the engine move a unit, and this knows which of those verbs
 * a given record means. Runs on the game thread, dispatched there by
 * {@link Orders#onGameThread} (wiki: issuing-orders).
 */
final class OrderDispatch {

    private OrderDispatch() {
    }

    /** Applies one parsed order. Runs on the game thread. */
    static void apply(CommandRecord command) {
        Object engine = EngineHandle.current();
        if (command.kind() == CommandRecord.Kind.POSTURE) {
            // A type's standing orders, not a unit's -- stored for the
            // reflex pass, no unit to resolve.
            Reflexes.set(
                    command.buildType(),
                    command.x(),
                    command.y(),
                    command.unitId() == 1L,
                    (int) command.targetId());
            Log.info(
                    "channel: posture "
                            + command.buildType()
                            + " reach="
                            + command.x()
                            + " kite="
                            + command.unitId()
                            + " floor="
                            + command.targetId());
            return;
        }
        Object unit = Perception.findOwnedById(engine, command.unitId());
        if (unit == null) {
            Log.error(
                    "channel: no owned unit with id "
                            + command.unitId()
                            + "; it may have died since the sample");
            return;
        }
        if (command.kind() == CommandRecord.Kind.MOVE) {
            Orders.moveTo(engine, unit, command.x(), command.y());
            Log.info(
                    "channel: move "
                            + command.unitId()
                            + " -> ("
                            + command.x()
                            + ", "
                            + command.y()
                            + ")");
            return;
        }
        if (command.kind() == CommandRecord.Kind.ATTACK_MOVE) {
            Orders.attackMoveTo(engine, unit, command.x(), command.y());
            Log.info(
                    "channel: attack-move "
                            + command.unitId()
                            + " -> ("
                            + command.x()
                            + ", "
                            + command.y()
                            + ")");
            return;
        }
        if (command.kind() == CommandRecord.Kind.ATTACK) {
            // The target is looked up among what is visible rather than what is
            // owned, and it is looked up now rather than trusted from the
            // sample: a target can die or slip back into fog between the
            // planner deciding and the order arriving, and an attack on a
            // stale identity is a command the engine would drop silently.
            Object target = Perception.findVisibleById(engine, command.targetId());
            if (target == null) {
                Log.error(
                        "channel: no visible unit with id "
                                + command.targetId()
                                + "; the target died or left sight since the sample");
                return;
            }
            Orders.attack(engine, unit, target);
            Log.info("channel: attack " + command.targetId() + " by " + command.unitId());
            return;
        }
        if (command.kind() == CommandRecord.Kind.PRODUCE) {
            Orders.produce(engine, unit, command.buildType());
            Log.info(
                    "channel: produce "
                            + command.buildType()
                            + " by "
                            + command.unitId());
            return;
        }
        if (command.kind() == CommandRecord.Kind.ABILITY) {
            Orders.ability(engine, unit, command.actionKey());
            Log.info(
                    "channel: ability '"
                            + command.actionKey()
                            + "' by "
                            + command.unitId());
            return;
        }
        if (command.kind() == CommandRecord.Kind.ABILITY_AT) {
            Orders.abilityAt(engine, unit, command.actionKey(), command.x(), command.y());
            Log.info(
                    "channel: ability '"
                            + command.actionKey()
                            + "' by "
                            + command.unitId()
                            + " at ("
                            + command.x()
                            + ", "
                            + command.y()
                            + ")");
            return;
        }
        Orders.buildAt(engine, unit, command.buildType(), command.x(), command.y());
        // Watched from the moment of dispatch: the engine refuses a blocked
        // placement by silently dropping the waypoint, and the watch is what
        // turns that silence into a refused record in the next sample.
        BuildWatch.record(
                command.unitId(),
                command.buildType(),
                command.x(),
                command.y(),
                EngineAccess.readIntField(engine, StateStream.FRAME_FIELD));
        Log.info(
                "channel: build "
                        + command.buildType()
                        + " by "
                        + command.unitId()
                        + " at ("
                        + command.x()
                        + ", "
                        + command.y()
                        + ")");
    }
}
