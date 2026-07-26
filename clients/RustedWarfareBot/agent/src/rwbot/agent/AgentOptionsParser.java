package rwbot.agent;

/**
 * Turns the {@code -javaagent:<jar>=<options>} argument into {@link AgentOptions}.
 *
 * <p>Format is {@code key=value} pairs separated by {@code ;}. An unrecognised
 * key is an error rather than a warning: a misspelled option that silently does
 * nothing would present as "the agent ignored me" during a probe, which is the
 * most expensive kind of quiet failure to chase.
 *
 * <p>Every value is validated at its own type — a port that is not a port, a
 * count that is not positive, an offset that is not two numbers — and each
 * rejection names the option that carried it. Nothing is coerced, and a default
 * applies only when a key is absent altogether, never past a malformed value.
 *
 * <p>Split out of the value type because the two change for different reasons:
 * adding an option is a change here, while what an option means is a change
 * there. The parsing is also the larger half, and it was crowding out the
 * answer to "what can the agent be asked to do".
 */
final class AgentOptionsParser {

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

    /** Default move offset: far enough that arrival is unambiguous, in world units. */
    private static final float[] DEFAULT_MOVE_BY = {240.0f, 0.0f};

    /** Default milliseconds between world samples: four decisions a second. */
    private static final int DEFAULT_SAMPLE_MS = 250;

    private AgentOptionsParser() {
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
