package rwbot.agent;

/**
 * Agent logging, prefixed so it is separable from engine output.
 *
 * <p>The engine captures its own log through {@code -log}; agent lines go to
 * the process streams and land interleaved in the same console capture. The
 * prefix is what lets {@code rw_bot.harness.boot_log} tell them apart.
 */
final class Log {

    private static final String PREFIX = "[rw-agent] ";

    private Log() {
    }

    static void info(String message) {
        System.out.println(prefixEveryLine(message, PREFIX));
        System.out.flush();
    }

    static void error(String message) {
        System.err.println(prefixEveryLine(message, PREFIX + "ERROR "));
        System.err.flush();
    }

    /**
     * Prefixes every line of a message, not merely the first.
     *
     * <p>Multi-line output is the normal case here -- a discovery snapshot is
     * hundreds of lines -- and the engine captures {@code System.out} into its
     * own log, interleaving agent output with engine output. A prefix on only
     * the first line leaves the rest indistinguishable from engine chatter and
     * unextractable by anything reading the log afterwards.
     *
     * @param message Text to prefix, possibly containing newlines.
     * @param prefix Prefix to apply to each line.
     * @return The prefixed text.
     */
    private static String prefixEveryLine(String message, String prefix) {
        return prefix + message.replace("\n", "\n" + prefix);
    }
}
