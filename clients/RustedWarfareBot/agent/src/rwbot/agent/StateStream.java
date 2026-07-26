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
 * sample and states how many records of each kind follow; each {@code entity}
 * record carries one visible entity and each {@code pool} record one visible
 * resource pool. Fields are limited to what has been verified against the
 * engine -- frame counter, millisecond clock, class, position -- rather than
 * everything reachable.
 *
 * <p>Pools are terrain rather than units and so appear in no entity list, but
 * they are carried in the same sample rather than sent once at connect time.
 * That keeps a sample self-contained, which is what lets a captured session
 * replay through the planner with nothing else alongside it.
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
        java.util.List<Object> visible = Perception.visibleEntities(engine);
        java.util.List<MapTiles.Pool> pools = MapTiles.visiblePools(engine);
        java.util.List<BuildOptions.Option> options = BuildOptions.ownedOptions(engine);
        int frame = EngineAccess.readIntField(engine, FRAME_FIELD);
        int clock = EngineAccess.readIntField(engine, CLOCK_FIELD);

        out.append(
                        frameRecord(
                                frame,
                                clock,
                                visible.size(),
                                pools.size(),
                                options.size(),
                                Perception.creditsOf(engine)))
                .append('\n');
        for (int index = 0; index < visible.size(); index++) {
            out.append(entityRecord(frame, index, visible.get(index), engine)).append('\n');
        }
        for (int index = 0; index < pools.size(); index++) {
            out.append(poolRecord(frame, index, pools.get(index))).append('\n');
        }
        for (int index = 0; index < options.size(); index++) {
            out.append(optionRecord(frame, index, options.get(index))).append('\n');
        }
        return out.toString();
    }

    /** Renders the record that opens a sample. */
    static String frameRecord(
            int frame,
            int clockMs,
            int visibleCount,
            int poolCount,
            int optionCount,
            int credits) {
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
        appendInt(out, "pools", poolCount);
        out.append(',');
        appendInt(out, "options", optionCount);
        out.append(',');
        appendInt(out, "credits", credits);
        out.append('}');
        return out.toString();
    }

    /**
     * Renders one thing an owned unit can make.
     *
     * <p>Addressed by the producing unit's engine id rather than its type,
     * because that id is what an order carries. A planner reading these needs
     * no separate table to get from "I want a tank" to "order unit 228".
     */
    static String optionRecord(int frame, int index, BuildOptions.Option option) {
        StringBuilder out = new StringBuilder();
        out.append('{');
        appendString(out, "kind", "option");
        out.append(',');
        appendInt(out, "frame", frame);
        out.append(',');
        appendInt(out, "index", index);
        out.append(',');
        appendLong(out, "unit_id", option.unitId());
        out.append(',');
        appendString(out, "produces", option.produces());
        out.append(',');
        appendInt(out, "action", option.actionIndex());
        out.append(',');
        appendBool(out, "placed", option.placed());
        out.append(',');
        appendBool(out, "available", option.available());
        out.append('}');
        return out.toString();
    }

    /**
     * Renders one visible resource pool.
     *
     * <p>Carries the tile coordinate and the world point at the tile's centre.
     * The tile coordinate is the pool's identity: it is integral, it never
     * moves, and it is the unit the engine's own placement check works in. The
     * world point is what a build order needs, because orders are addressed in
     * world space.
     */
    static String poolRecord(int frame, int index, MapTiles.Pool pool) {
        StringBuilder out = new StringBuilder();
        out.append('{');
        appendString(out, "kind", "pool");
        out.append(',');
        appendInt(out, "frame", frame);
        out.append(',');
        appendInt(out, "index", index);
        out.append(',');
        appendInt(out, "tile_x", pool.tileX());
        out.append(',');
        appendInt(out, "tile_y", pool.tileY());
        out.append(',');
        appendFloat(out, "x", pool.x());
        out.append(',');
        appendFloat(out, "y", pool.y());
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
        float[] at = Perception.positionOf(entity);
        float[] health = Perception.healthOf(entity);
        StringBuilder out = new StringBuilder();
        out.append('{');
        appendString(out, "kind", "entity");
        out.append(',');
        appendInt(out, "frame", frame);
        out.append(',');
        appendInt(out, "index", index);
        out.append(',');
        appendLong(out, "id", Perception.idOf(entity));
        out.append(',');
        appendString(out, "type", Perception.typeNameOf(entity));
        out.append(',');
        appendString(out, "class", entity.getClass().getName());
        out.append(',');
        appendFloat(out, "x", at[0]);
        out.append(',');
        appendFloat(out, "y", at[1]);
        out.append(',');
        appendInt(out, "team", Perception.teamOf(entity));
        out.append(',');
        appendBool(out, "mine", Perception.isOwnedByLocalPlayer(engine, entity));
        out.append(',');
        appendFloat(out, "hp", health[0]);
        out.append(',');
        appendFloat(out, "max_hp", health[1]);
        out.append(',');
        appendBool(out, "complete", Perception.isComplete(entity));
        out.append(',');
        appendInt(out, "queued", Perception.queuedCountOf(entity));
        out.append('}');
        return out.toString();
    }

    private static void appendBool(StringBuilder out, String key, boolean value) {
        Json.quote(out, key);
        out.append(':').append(value);
    }

    /** Appends a double, rejecting non-finite values for the same reason as floats. */
    private static void appendDouble(StringBuilder out, String key, double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalStateException(
                    "rw-agent: refusing to serialise non-finite " + key + "=" + value);
        }
        Json.quote(out, key);
        out.append(':').append(value);
    }

    private static void appendString(StringBuilder out, String key, String value) {
        Json.quote(out, key);
        out.append(':');
        Json.quote(out, value);
    }

    private static void appendLong(StringBuilder out, String key, long value) {
        Json.quote(out, key);
        out.append(':').append(value);
    }

    private static void appendInt(StringBuilder out, String key, int value) {
        Json.quote(out, key);
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
        Json.quote(out, key);
        out.append(':').append(value);
    }
}
