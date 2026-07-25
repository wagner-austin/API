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
        System.out.println(PREFIX + message);
        System.out.flush();
    }

    static void error(String message) {
        System.err.println(PREFIX + "ERROR " + message);
        System.err.flush();
    }
}
