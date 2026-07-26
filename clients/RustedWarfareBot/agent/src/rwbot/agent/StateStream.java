package rwbot.agent;

/**
 * Serialises the live world as newline-delimited JSON.
 *
 * <p>This is the agent half of the wire contract. A JSONL stream is itself a
 * corpus: a captured session replays through the planner offline with no game
 * running, which is why the format was chosen for replayability ahead of
 * performance (wiki: runtime-split-java-agent-python-brain).
 *
 * <p><b>Every line is a flat object of scalar values.</b> No nesting, no
 * arrays. That is a deliberate constraint on the producer rather than an
 * accident of what was easy: the Python reader runs under
 * {@code disallow_any_expr}, where {@code json.loads} is unusable because its
 * return value is {@code Any} and suppressions are banned. A flat scalar object
 * is parseable strictly into typed fields, so constraining the format here is
 * what lets the consumer stay fully typed.
 *
 * <p>Records are discriminated by {@code kind}. A {@code frame} record opens a
 * sample and states how many entity records follow; each {@code entity} record
 * carries one owned entity. Fields are limited to what has been verified
 * against the engine -- frame counter, millisecond clock, class, position --
 * rather than everything reachable.
 */
final class StateStream {

    private static final String FRAME_FIELD = "bx";
    private static final String CLOCK_FIELD = "by";

    private StateStream() {
    }

    /**
     * Renders one sample of the world as NDJSON.
     *
     * <p>Reads run on the game thread, because a position sampled mid-tick can
     * be torn exactly as a write can corrupt (wiki: issuing-orders).
     *
     * @param engine The live engine instance.
     * @return One frame record followed by one record per visible entity, each
     *     newline-terminated.
     */
    static String sample(Object engine) {
        StringBuilder out = new StringBuilder();
        java.util.List<Object> visible = Orders.visibleEntities(engine);
        int frame = Orders.readIntField(engine, FRAME_FIELD);
        int clock = Orders.readIntField(engine, CLOCK_FIELD);

        out.append(frameRecord(frame, clock, visible.size(), Orders.creditsOf(engine)))
                .append('\n');
        for (int index = 0; index < visible.size(); index++) {
            out.append(entityRecord(frame, index, visible.get(index), engine)).append('\n');
        }
        return out.toString();
    }

    /** Renders the record that opens a sample. */
    static String frameRecord(int frame, int clockMs, int visibleCount, int credits) {
        StringBuilder out = new StringBuilder();
        out.append('{');
        appendString(out, "kind", "frame");
        out.append(',');
        appendInt(out, "frame", frame);
        out.append(',');
        appendInt(out, "clock_ms", clockMs);
        out.append(',');
        appendInt(out, "visible", visibleCount);
        out.append(',');
        appendInt(out, "credits", credits);
        out.append('}');
        return out.toString();
    }

    /**
     * Renders one owned entity.
     *
     * <p>Carries both {@code index} and {@code id}. Index is enumeration order
     * and is useful only for reading a single sample; it renumbers whenever
     * anything is built or dies. {@code id} is the engine's own object
     * identity, assigned once at construction, and is the handle an order is
     * dispatched against.
     */
    static String entityRecord(int frame, int index, Object entity, Object engine) {
        float[] at = Orders.positionOf(entity);
        float[] health = Orders.healthOf(entity);
        StringBuilder out = new StringBuilder();
        out.append('{');
        appendString(out, "kind", "entity");
        out.append(',');
        appendInt(out, "frame", frame);
        out.append(',');
        appendInt(out, "index", index);
        out.append(',');
        appendLong(out, "id", Orders.idOf(entity));
        out.append(',');
        appendString(out, "type", Orders.typeNameOf(entity));
        out.append(',');
        appendString(out, "class", entity.getClass().getName());
        out.append(',');
        appendFloat(out, "x", at[0]);
        out.append(',');
        appendFloat(out, "y", at[1]);
        out.append(',');
        appendInt(out, "team", Orders.teamOf(entity));
        out.append(',');
        appendBool(out, "mine", Orders.isOwnedByLocalPlayer(engine, entity));
        out.append(',');
        appendFloat(out, "hp", health[0]);
        out.append(',');
        appendFloat(out, "max_hp", health[1]);
        out.append('}');
        return out.toString();
    }

    private static void appendBool(StringBuilder out, String key, boolean value) {
        quote(out, key);
        out.append(':').append(value);
    }

    /** Appends a double, rejecting non-finite values for the same reason as floats. */
    private static void appendDouble(StringBuilder out, String key, double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalStateException(
                    "rw-agent: refusing to serialise non-finite " + key + "=" + value);
        }
        quote(out, key);
        out.append(':').append(value);
    }

    private static void appendString(StringBuilder out, String key, String value) {
        quote(out, key);
        out.append(':');
        quote(out, value);
    }

    private static void appendLong(StringBuilder out, String key, long value) {
        quote(out, key);
        out.append(':').append(value);
    }

    private static void appendInt(StringBuilder out, String key, int value) {
        quote(out, key);
        out.append(':').append(value);
    }

    /**
     * Appends a float.
     *
     * <p>Non-finite values are rejected rather than written: JSON has no
     * encoding for them, and emitting a bare {@code NaN} would produce a stream
     * that only a lenient parser accepts. A non-finite coordinate means the
     * read was wrong, so it fails here rather than downstream.
     */
    private static void appendFloat(StringBuilder out, String key, float value) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            throw new IllegalStateException(
                    "rw-agent: refusing to serialise non-finite " + key + "=" + value);
        }
        quote(out, key);
        out.append(':').append(value);
    }

    /** Writes a JSON string, escaping what the grammar requires. */
    private static void quote(StringBuilder out, String text) {
        out.append('"');
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                    break;
            }
        }
        out.append('"');
    }
}
