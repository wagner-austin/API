package rwbot.agent;

/**
 * The assertion vocabulary the agent's self-checks share.
 *
 * <p>One line per check, printed as it runs, so a failing gate names the
 * property that broke rather than a line number. Kept separate from the check
 * groups themselves because every group needs it and none of them owns it.
 */
final class Check {

    private Check() {
    }

    /** Reports one assertion, returning 1 when it failed. */
    static int expect(boolean condition, String description) {
        if (condition) {
            System.out.println("ok   " + description);
            return 0;
        }
        System.out.println("FAIL " + description);
        return 1;
    }
}
