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

    private final int[] discoverAtSeconds;
    private final boolean exitAfterDiscovery;

    private AgentOptions(int[] discoverAtSeconds, boolean exitAfterDiscovery) {
        this.discoverAtSeconds = discoverAtSeconds;
        this.exitAfterDiscovery = exitAfterDiscovery;
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
            return new AgentOptions(new int[0], false);
        }

        int[] discoverAt = new int[0];
        boolean exitAfter = false;
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
            } else {
                throw new IllegalArgumentException(
                        "unknown agent option " + key + "; supported: " + DISCOVER_AT + ", "
                                + EXIT_AFTER);
            }
        }
        return new AgentOptions(discoverAt, exitAfter);
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
