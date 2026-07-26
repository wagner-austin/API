package rwbot.agent;

/**
 * One order arriving from the planner, parsed from a single NDJSON line.
 *
 * <p>The parser is deliberately strict and deliberately small. It accepts a
 * flat object of scalar values and nothing else: no nesting, no arrays, no
 * escapes beyond the six JSON defines. That is the same constraint
 * {@link StateStream} places on the outbound direction, and for the same
 * reason -- a format narrow enough to parse exactly is a format neither side
 * can drift on.
 *
 * <p>Every rejection is loud. A command that cannot be parsed is not skipped:
 * a planner whose orders are silently dropped looks identical to a planner
 * whose orders do nothing, and that failure has already cost this project two
 * runs (wiki: issuing-orders, building-structures).
 */
final class CommandRecord {

    /** Order verbs the agent accepts. */
    enum Kind {
        /** Send a unit to a world position. */
        MOVE,
        /** Have a builder place a structure at a world position. */
        BUILD,

        /**
         * Queue a unit in a building that makes it.
         *
         * <p>Carries no position, because there is nowhere to choose: the unit
         * appears at the building that made it (wiki: mechanics-build-actions).
         */
        PRODUCE,

        /**
         * Send a unit to attack another unit.
         *
         * <p>Addresses the target by engine identity rather than by position,
         * which is the whole difference from a move. A target that walks away
         * is still the target; a move to where it stood is a march to empty
         * ground (wiki: policy-combat).
         */
        ATTACK
    }

    private final Kind kind;
    private final long unitId;
    private final float x;
    private final float y;
    private final String buildType;
    private final long targetId;

    private CommandRecord(
            Kind kind, long unitId, float x, float y, String buildType, long targetId) {
        this.kind = kind;
        this.unitId = unitId;
        this.x = x;
        this.y = y;
        this.buildType = buildType;
        this.targetId = targetId;
    }

    Kind kind() {
        return kind;
    }

    /** Engine object identity of the unit to order. */
    long unitId() {
        return unitId;
    }

    float x() {
        return x;
    }

    float y() {
        return y;
    }

    /** Unit-type name to build; empty for a move. */
    /** Engine object identity of the unit to attack. Zero unless ATTACK. */
    long targetId() {
        return targetId;
    }

    String buildType() {
        return buildType;
    }

    /**
     * Parses one command line.
     *
     * @param line A single NDJSON object, without its newline.
     * @return The parsed command.
     * @throws IllegalArgumentException When the line is not a flat object, a
     *     required field is absent, a field has the wrong shape, or the verb is
     *     unknown.
     */
    static CommandRecord parse(String line) {
        java.util.Map<String, String> fields = Json.flatObject(line);

        String kindText = require(fields, "kind", line);
        long unitId = requireLong(fields, "unit_id", line);

        // Required per verb rather than up front. A produce command has no
        // position to carry, and demanding one would force the sender to
        // invent a coordinate that nothing reads.
        if ("move".equals(kindText)) {
            reject(fields, "type", line);
            return new CommandRecord(
                    Kind.MOVE,
                    unitId,
                    requireFloat(fields, "x", line),
                    requireFloat(fields, "y", line),
                    "",
                    0L);
        }
        if ("build".equals(kindText)) {
            return new CommandRecord(
                    Kind.BUILD,
                    unitId,
                    requireFloat(fields, "x", line),
                    requireFloat(fields, "y", line),
                    requireType(fields, "build", line),
                    0L);
        }
        if ("produce".equals(kindText)) {
            reject(fields, "x", line);
            reject(fields, "y", line);
            return new CommandRecord(
                    Kind.PRODUCE, unitId, 0.0f, 0.0f, requireType(fields, "produce", line), 0L);
        }
        if ("attack".equals(kindText)) {
            // No position and no type: the target's identity is the whole of
            // the order, and a coordinate here would be a number nothing reads.
            reject(fields, "x", line);
            reject(fields, "y", line);
            reject(fields, "type", line);
            return new CommandRecord(
                    Kind.ATTACK, unitId, 0.0f, 0.0f, "", requireLong(fields, "target_id", line));
        }
        throw new IllegalArgumentException(
                "unknown command kind '" + kindText
                        + "'; expected move, build, produce or attack: " + line);
    }

    /** Reads the unit-type field, which no verb that carries it may leave blank. */
    private static String requireType(
            java.util.Map<String, String> fields, String verb, String line) {
        String type = require(fields, "type", line);
        if (type.isEmpty()) {
            throw new IllegalArgumentException(verb + " command has an empty type: " + line);
        }
        return type;
    }

    private static String require(java.util.Map<String, String> fields, String key, String line) {
        String value = fields.get(key);
        if (value == null) {
            throw new IllegalArgumentException("command is missing '" + key + "': " + line);
        }
        return value;
    }

    /** Rejects a field that does not belong to this verb, rather than ignoring it. */
    private static void reject(java.util.Map<String, String> fields, String key, String line) {
        if (fields.containsKey(key)) {
            throw new IllegalArgumentException(
                    "move command must not carry '" + key + "': " + line);
        }
    }

    private static long requireLong(
            java.util.Map<String, String> fields, String key, String line) {
        String text = require(fields, key, line);
        try {
            return Long.parseLong(text);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "command field '" + key + "' is not a whole number: " + line, e);
        }
    }

    private static float requireFloat(
            java.util.Map<String, String> fields, String key, String line) {
        String text = require(fields, key, line);
        float value;
        try {
            value = Float.parseFloat(text);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "command field '" + key + "' is not a number: " + line, e);
        }
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            throw new IllegalArgumentException(
                    "command field '" + key + "' is not finite: " + line);
        }
        return value;
    }
}
