package rwbot.agent;

/**
 * Options passed to the agent through the {@code -javaagent:<jar>=<options>}
 * argument.
 *
 * <p>Format is {@code key=value} pairs separated by {@code ;}. An unrecognised
 * key is an error rather than a warning: a misspelled option that silently does
 * nothing would present as "the agent ignored me" during a probe, which is the
 * most expensive kind of quiet failure to chase.
 */
final class AgentOptions {

    private static final String DISCOVER_AT = "discoverAtSeconds";
    private static final String EXIT_AFTER = "exitAfterDiscovery";
    private static final String INSPECT_FIELDS = "inspectFields";
    private static final String FIND_UNDER = "findElementsUnder";
    private static final String STATE_OUT = "stateOutPath";
    private static final String TYPE_FLAGS_OUT = "typeFlagsPath";
    private static final String ORDER_AT = "orderMoveAtSeconds";
    private static final String ORDER_BY = "orderMoveBy";
    private static final String ORDER_INDEX = "orderMoveUnitIndex";
    private static final String BUILD_TYPE = "buildType";
    private static final String CHANNEL_PORT = "channelPort";
    private static final String SAMPLE_MS = "sampleIntervalMs";

    private final int[] discoverAtSeconds;
    private final boolean exitAfterDiscovery;
    private final String[] inspectFields;
    private final String findElementsUnder;
    private final String stateOutPath;
    private final String typeFlagsPath;
    private final int orderMoveAtSeconds;
    private final float[] orderMoveBy;
    private final int orderMoveUnitIndex;
    private final String buildType;
    private final int channelPort;
    private final int sampleIntervalMs;

    private AgentOptions(
            int[] discoverAtSeconds,
            boolean exitAfterDiscovery,
            String[] inspectFields,
            String findElementsUnder,
            int orderMoveAtSeconds,
            float[] orderMoveBy,
            int orderMoveUnitIndex,
            String buildType,
            String stateOutPath,
            String typeFlagsPath,
            int channelPort,
            int sampleIntervalMs) {
        this.discoverAtSeconds = discoverAtSeconds;
        this.exitAfterDiscovery = exitAfterDiscovery;
        this.inspectFields = inspectFields;
        this.findElementsUnder = findElementsUnder;
        this.orderMoveAtSeconds = orderMoveAtSeconds;
        this.orderMoveBy = orderMoveBy;
        this.orderMoveUnitIndex = orderMoveUnitIndex;
        this.buildType = buildType;
        this.stateOutPath = stateOutPath;
        this.typeFlagsPath = typeFlagsPath;
        this.channelPort = channelPort;
        this.sampleIntervalMs = sampleIntervalMs;
    }

    /**
     * Loopback port the planner connects to, or 0 when the channel is off.
     *
     * <p>Off by default. A probe run and a planner-driven run are different
     * things, and opening a listening socket as a side effect of attaching the
     * agent would make every discovery run also a server.
     */
    int channelPort() {
        return channelPort;
    }

    /** True when a planner channel was requested. */
    boolean channelRequested() {
        return channelPort > 0;
    }

    /**
     * Milliseconds between world samples pushed to the planner.
     *
     * <p>Defaults to 250 ms -- four decisions a second. An RTS does not need
     * per-tick decisions, and the sampling rate is what keeps a cross-process
     * planner viable (wiki: runtime-split-java-agent-python-brain).
     */
    int sampleIntervalMs() {
        return sampleIntervalMs;
    }

    /**
     * Unit-type name to construct instead of moving, or empty to move.
     *
     * <p>Shares the timer and roster index with the move probe because the two
     * differ only in the verb: same subject, same destination, different
     * command setter. Keeping them one option set makes that similarity
     * visible rather than duplicating the scheduling.
     */
    String buildType() {
        return buildType;
    }

    /**
     * Index into the owned-entity roster of the unit to order.
     *
     * <p>The agent does not decide which unit is worth moving, or even which
     * can move: it publishes the roster and dispatches against an index.
     * Choosing is the planner's job, and a mobility predicate guessed here
     * would be exactly the decision logic the agent must not hold (wiki:
     * multiplayer-portability-invariants).
     */
    int orderMoveUnitIndex() {
        return orderMoveUnitIndex;
    }

    /**
     * Elapsed time at which to order the player's first unit to move, or 0 when
     * not requested.
     *
     * <p>A time rather than a flag because the order is only meaningful once a
     * map has loaded and the player owns something; issuing at boot would
     * report "no unit" and prove nothing.
     */
    int orderMoveAtSeconds() {
        return orderMoveAtSeconds;
    }

    /** True when a move order was requested. */
    boolean orderRequested() {
        return orderMoveAtSeconds > 0;
    }

    /**
     * World-space offset the ordered unit is sent by, as {x, y}.
     *
     * <p>An offset rather than an absolute point: the destination has to be
     * somewhere the unit can actually reach, and the only position known to be
     * on reachable terrain is the one the unit already occupies.
     */
    float[] orderMoveBy() {
        return orderMoveBy.clone();
    }

    /**
     * Binary-name prefix to search the live object graph for, or empty when not
     * requested.
     */
    String findElementsUnder() {
        return findElementsUnder;
    }

    /**
     * Absolute path the NDJSON world stream is appended to, or empty when not
     * requested. Written at each discovery offset, so it shares that schedule
     * rather than inventing a second one.
     */
    String stateOutPath() {
        return stateOutPath;
    }

    /**
     * Absolute path the unit-type placement flags are written to, or empty when
     * not requested.
     *
     * <p>Written once, at the first discovery offset, because the answer cannot
     * change during a match: unit types are loaded with the assets and the mod
     * set is fixed at boot. Shares the discovery schedule for the same reason
     * the state dump does -- both need a loaded game and neither needs its own
     * timer.
     */
    String typeFlagsPath() {
        return typeFlagsPath;
    }

    /**
     * Engine field names whose elements to expand, in declaration order.
     *
     * <p>A whole-object snapshot reports that a collection holds eleven things;
     * it cannot say what distinguishes them. Naming a field opts into one extra
     * level of depth for that field only, which keeps the default snapshot
     * cheap and the expensive read deliberate.
     */
    String[] inspectFields() {
        return inspectFields.clone();
    }

    /**
     * Whether to halt the JVM once the last snapshot has been emitted.
     *
     * <p>Off by default, and named rather than implied: a probe that killed the
     * game as a side effect of asking for a snapshot would be a surprise the
     * first time someone combined discovery with a longer run.
     */
    boolean exitAfterDiscovery() {
        return exitAfterDiscovery;
    }

    /**
     * Elapsed times, in seconds, at which to dump a discovery snapshot.
     *
     * <p>Empty when discovery was not requested. Multiple times are the point:
     * comparing a snapshot taken before a map loads with one taken after
     * identifies which fields hold match state, which a single dump cannot.
     */
    int[] discoverAtSeconds() {
        return discoverAtSeconds.clone();
    }

    /** True when any discovery snapshot was requested. */
    boolean discoveryRequested() {
        return discoverAtSeconds.length > 0;
    }

    /**
     * Parses the agent argument string.
     *
     * @param argument The raw {@code -javaagent} argument, or null when the
     *     agent was attached with no {@code =} suffix.
     * @return The parsed options.
     * @throws IllegalArgumentException When a key is unrecognised, a pair is
     *     malformed, or a time is not a positive integer.
     */
    static AgentOptions parse(String argument) {
        if (argument == null || argument.trim().isEmpty()) {
            return new AgentOptions(
                    new int[0],
                    false,
                    new String[0],
                    "",
                    0,
                    DEFAULT_MOVE_BY.clone(),
                    0,
                    "",
                    "",
                    "",
                    0,
                    DEFAULT_SAMPLE_MS);
        }

        int[] discoverAt = new int[0];
        boolean exitAfter = false;
        String[] inspect = new String[0];
        String findUnder = "";
        String stateOut = "";
        String typeFlagsOut = "";
        int orderAt = 0;
        float[] orderBy = DEFAULT_MOVE_BY.clone();
        int orderIndex = 0;
        String buildType = "";
        int channelPort = 0;
        int sampleIntervalMs = DEFAULT_SAMPLE_MS;
        for (String pair : argument.split(";")) {
            String trimmed = pair.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            int equals = trimmed.indexOf('=');
            if (equals < 1) {
                throw new IllegalArgumentException(
                        "malformed agent option " + trimmed + "; expected key=value");
            }
            String key = trimmed.substring(0, equals).trim();
            String value = trimmed.substring(equals + 1).trim();
            if (DISCOVER_AT.equals(key)) {
                discoverAt = parseSeconds(value);
            } else if (EXIT_AFTER.equals(key)) {
                exitAfter = parseBoolean(value);
            } else if (INSPECT_FIELDS.equals(key)) {
                inspect = parseNames(value);
            } else if (STATE_OUT.equals(key)) {
                if (value.isEmpty()) {
                    throw new IllegalArgumentException(STATE_OUT + " expects a path");
                }
                stateOut = value;
            } else if (TYPE_FLAGS_OUT.equals(key)) {
                if (value.isEmpty()) {
                    throw new IllegalArgumentException(TYPE_FLAGS_OUT + " expects a path");
                }
                typeFlagsOut = value;
            } else if (FIND_UNDER.equals(key)) {
                if (value.isEmpty()) {
                    throw new IllegalArgumentException(FIND_UNDER + " expects a package prefix");
                }
                findUnder = value;
            } else if (ORDER_AT.equals(key)) {
                orderAt = parseOneSecond(value);
            } else if (ORDER_BY.equals(key)) {
                orderBy = parseOffset(value);
            } else if (ORDER_INDEX.equals(key)) {
                orderIndex = parseIndex(value);
            } else if (BUILD_TYPE.equals(key)) {
                if (value.isEmpty()) {
                    throw new IllegalArgumentException(BUILD_TYPE + " expects a unit-type name");
                }
                buildType = value;
            } else if (CHANNEL_PORT.equals(key)) {
                channelPort = parsePort(value);
            } else if (SAMPLE_MS.equals(key)) {
                sampleIntervalMs = parseInterval(value);
            } else {
                throw new IllegalArgumentException(
                        "unknown agent option " + key + "; supported: " + DISCOVER_AT + ", "
                                + EXIT_AFTER + ", " + INSPECT_FIELDS + ", " + FIND_UNDER + ", " + STATE_OUT + ", "
                                + TYPE_FLAGS_OUT + ", "
                                + ORDER_AT + ", " + ORDER_BY + ", " + ORDER_INDEX + ", " + BUILD_TYPE + ", " + CHANNEL_PORT + ", "
                                + SAMPLE_MS);
            }
        }
        return new AgentOptions(
                discoverAt,
                exitAfter,
                inspect,
                findUnder,
                orderAt,
                orderBy,
                orderIndex,
                buildType,
                stateOut,
                typeFlagsOut,
                channelPort,
                sampleIntervalMs);
    }

    /** Default move offset: far enough that arrival is unambiguous, in world units. */
    private static final float[] DEFAULT_MOVE_BY = {240.0f, 0.0f};

    /** Parses a single positive whole-second offset. */
    private static int parseOneSecond(String value) {
        int parsed;
        try {
            parsed = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(ORDER_AT + " expects whole seconds, got " + value, e);
        }
        if (parsed <= 0) {
            throw new IllegalArgumentException(ORDER_AT + " expects positive seconds, got " + parsed);
        }
        return parsed;
    }

    /** Default milliseconds between world samples: four decisions a second. */
    private static final int DEFAULT_SAMPLE_MS = 250;

    /** Parses a TCP port in the unprivileged range. */
    private static int parsePort(String value) {
        int parsed;
        try {
            parsed = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(CHANNEL_PORT + " expects a port, got " + value, e);
        }
        if (parsed < 1024 || parsed > 65535) {
            throw new IllegalArgumentException(
                    CHANNEL_PORT + " expects 1024-65535, got " + parsed);
        }
        return parsed;
    }

    /** Parses a positive sample interval in milliseconds. */
    private static int parseInterval(String value) {
        int parsed;
        try {
            parsed = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    SAMPLE_MS + " expects whole milliseconds, got " + value, e);
        }
        if (parsed <= 0) {
            throw new IllegalArgumentException(
                    SAMPLE_MS + " expects a positive interval, got " + parsed);
        }
        return parsed;
    }

    /** Parses a non-negative roster index. */
    private static int parseIndex(String value) {
        int parsed;
        try {
            parsed = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(ORDER_INDEX + " expects an integer, got " + value, e);
        }
        if (parsed < 0) {
            throw new IllegalArgumentException(
                    ORDER_INDEX + " expects a non-negative index, got " + parsed);
        }
        return parsed;
    }

    /** Parses an {@code x,y} world-space offset. */
    private static float[] parseOffset(String value) {
        String[] parts = value.split(",");
        if (parts.length != 2) {
            throw new IllegalArgumentException(ORDER_BY + " expects x,y, got " + value);
        }
        float[] offset = new float[2];
        for (int i = 0; i < 2; i++) {
            try {
                offset[i] = Float.parseFloat(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        ORDER_BY + " expects two numbers, got " + value, e);
            }
        }
        return offset;
    }

    /** Parses a comma-separated list of non-blank field names. */
    private static String[] parseNames(String value) {
        String[] parts = value.split(",");
        String[] names = new String[parts.length];
        for (int i = 0; i < parts.length; i++) {
            names[i] = parts[i].trim();
            if (names[i].isEmpty()) {
                throw new IllegalArgumentException(
                        INSPECT_FIELDS + " expects non-blank field names, got " + value);
            }
        }
        return names;
    }

    /** Parses a strict boolean; anything else is an error rather than false. */
    private static boolean parseBoolean(String value) {
        if ("true".equals(value)) {
            return true;
        }
        if ("false".equals(value)) {
            return false;
        }
        throw new IllegalArgumentException(
                EXIT_AFTER + " expects true or false, got " + value);
    }

    /** Parses a comma-separated list of positive second offsets. */
    private static int[] parseSeconds(String value) {
        String[] parts = value.split(",");
        int[] seconds = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            int parsed;
            try {
                parsed = Integer.parseInt(part);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        DISCOVER_AT + " expects whole seconds, got " + part, e);
            }
            if (parsed <= 0) {
                throw new IllegalArgumentException(
                        DISCOVER_AT + " expects positive seconds, got " + parsed);
            }
            seconds[i] = parsed;
        }
        java.util.Arrays.sort(seconds);
        return seconds;
    }
}
