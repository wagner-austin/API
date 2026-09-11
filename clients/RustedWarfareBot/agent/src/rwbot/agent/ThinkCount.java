package rwbot.agent;

/**
 * Per-tick invocation counts of the AI's think entry points -- the last
 * discriminant the determinism arc named and never built.
 *
 * <p>Every prior instrument counted DRAWS ({@link RandomTap}) or read STATE
 * ({@link AiCadence}): both proved identical across byte-identical twins
 * whose worlds nonetheless forked, because a think pass can run and find
 * nothing -- no draw, no state change -- and only the invocation count
 * separates "ran and found nothing" from "never ran" (wiki:
 * policy-determinism, the 2026-09-06 pool-scan entry, and dettwin96: 48
 * same-difficulty twin pairs, zero identical, co-location irrelevant). The
 * wall-decoupling program starts here: two twins' per-tick think counts,
 * lined up, either step apart at a tick -- naming the scheduler quantity
 * the wall paces -- or match through a fork, which pushes the seam into
 * the think's own inputs.
 *
 * <p>Four counters, one per patched entry ({@link ThinkCountTransformer}):
 * the task re-aim ({@code a.n.f()V}, the frame-75 fork's proven event), the
 * AI evaluation ({@code a.a.n(F)V}, the dettap8 burst driver), and the
 * group's two delta-fed drivers ({@code a.i.b(F)V}, {@code a.i.d(F)V}, the
 * tick-33 fork's chain). Static ints bumped by injected {@code invokestatic}
 * at each method's entry: no allocation, no draw, no branch -- an instrument
 * that touches its subject's stream is not an instrument (the RandomLedger
 * describe() lesson, wiki log 2026-09-07).
 *
 * <p>Counts are cumulative; readers difference them. Reset belongs to match
 * start alone ({@link MatchSetup}), so a menu world's thinks cannot shadow
 * the match's own.
 */
public final class ThinkCount {

    private static int aims;
    private static int evaluations;
    private static int groupThinks;
    private static int groupDrives;

    private ThinkCount() {
    }

    /**
     * Bumped at {@code a.n.f()V} entry: the task re-aim.
     *
     * <p>Public like {@link SideDraw}'s hooks and for the same measured
     * reason: the callers are the GAME's classes in their own package, and
     * an injected {@code invokestatic} of a package-private method throws
     * {@code IllegalAccessError} at first execution -- which the verifier
     * selftest cannot see, because access resolves at execution, not at
     * verification (the tca/tcb pair died exactly there, 2026-09-11).
     */
    public static void aim() {
        aims++;
    }

    /** Bumped at {@code a.a.n(F)V} entry: the AI evaluation pass. */
    public static void evaluate() {
        evaluations++;
    }

    /** Bumped at {@code a.i.b(F)V} entry: the group think driver. */
    public static void groupThink() {
        groupThinks++;
    }

    /** Bumped at {@code a.i.d(F)V} entry: the group update driver. */
    public static void groupDrive() {
        groupDrives++;
    }

    /**
     * Renders the four cumulative counts, compactly, for the cadence lines.
     *
     * @return {@code aim.eval.gthink.gdrive} as dot-joined counts.
     */
    static String describe() {
        return aims + "." + evaluations + "." + groupThinks + "." + groupDrives;
    }

    /**
     * Zeroes all four counters, at match start only: the menu background is
     * a running mission demo whose AI thinks too, and a cumulative count
     * that started before the seed was applied would difference wrongly
     * across the boundary the whole instrument exists to watch.
     */
    static void reset() {
        aims = 0;
        evaluations = 0;
        groupThinks = 0;
        groupDrives = 0;
    }
}
