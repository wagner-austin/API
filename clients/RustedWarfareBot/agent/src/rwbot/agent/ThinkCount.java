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
     * Zeroes all four counters and arms the scan budget, at match start
     * only: the menu background is a running mission demo whose AI thinks
     * AND scans too, so a cumulative count that started before the seed
     * was applied would difference wrongly across the boundary the whole
     * instrument exists to watch, and an earlier-armed scan budget would
     * be spent before the window the fork lives in.
     */
    static void reset() {
        aims = 0;
        evaluations = 0;
        groupThinks = 0;
        groupDrives = 0;
        scansRemaining = 48;
    }

    /**
     * Candidate-scan invocations still to log; zero until the match-start
     * reset. The fork lands within the opening scans, and forty-eight
     * covers the whole contested window with margin.
     */
    private static int scansRemaining;

    /**
     * The ArrayList the candidate scan iterates ({@code n.a(Z)} loops
     * {@code this.R.bn}) -- NOT {@code EngineNames.AI_SUBCONTROLLERS}, a
     * queue whose sizes merely track it; the first scan logger read the
     * queue under this label (wiki log 2026-09-11). Here rather than in
     * {@link EngineNames} by that table's own split-by-reader rule: this
     * class is the name's one reader, and the table sits at its ceiling.
     */
    private static final String AI_SCAN_ROSTER = "bn";

    /**
     * The scan's candidate class: {@code n.a(Z)} skips non-instances, and
     * its eligibility {@code getfield}s name it as owner -- the obfuscated
     * subclasses redeclare the same letters with different types, so a
     * reader resolving from {@code getClass()} prints shadows (wiki log
     * 2026-09-11). Placed like {@link #AI_SCAN_ROSTER}, and pinned to
     * 1.15 (code 176, build #28) like everything in {@link EngineNames}.
     */
    private static final String SCAN_CANDIDATE_CLASS = "com.corrodinggames.rts.game.a.i";

    /**
     * Logs the candidate scan's inputs at the moment it reads them -- the
     * mid-tick state every boundary snapshot was blind to.
     *
     * <p>The counter's own first pair pinned the fork to one branch of the
     * scan {@code a.n.a(boolean)}: eligibility ({@code i3.s && bl2 ||
     * i3.b != j.c}) judged differently on state the aitick lines printed
     * as identical, with the {@code s}-writer proven never to have run
     * (wiki log 2026-09-11). So this hook rides the scan's own entry,
     * receiver in hand, and prints each sub-controller's class, its
     * {@code b}/{@code c} category enums, and the contact flag {@code s}
     * the roster line never carried. Reads only, no draws, bounded.
     *
     * @param task The {@code a.n} task about to scan its AI's roster.
     */
    public static void scan(Object task) {
        if (scansRemaining <= 0) {
            return;
        }
        scansRemaining--;
        StringBuilder out = new StringBuilder("thinkscan");
        Object ai = readInherited(task, "R");
        Object held = ai == null ? null : readInherited(ai, AI_SCAN_ROSTER);
        java.util.Collection<?> roster = held == null ? null : ObjectView.containedValues(held);
        out.append(" bn=").append(roster == null ? "?" : Integer.valueOf(roster.size()));
        if (roster != null) {
            out.append(" [");
            boolean first = true;
            for (Object controller : roster) {
                if (controller == null) {
                    continue;
                }
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append(controller.getClass().getSimpleName());
                Class<?> candidate = candidateClass(controller);
                if (candidate == null) {
                    // The scan's own instanceof skips this entry, and its
                    // redeclared one-letter fields are shadows a reader must
                    // not print as the candidate state they are not.
                    out.append(":skip");
                    continue;
                }
                out.append(':')
                        .append(render(readDeclared(candidate, controller, "b")))
                        .append('/')
                        .append(render(readDeclared(candidate, controller, "c")))
                        .append("/s=")
                        .append(render(readDeclared(candidate, controller, "s")));
            }
            out.append(']');
        }
        Log.info(out.toString());
    }

    /**
     * Resolves the scan's candidate class in the controller's hierarchy, or
     * null when the controller is not a candidate -- exactly the entries
     * the scan's own {@code instanceof} skips.
     *
     * @param controller One roster entry.
     * @return The class whose declared fields the scan's eligibility reads
     *     name as owner, or null.
     */
    private static Class<?> candidateClass(Object controller) {
        for (Class<?> cls = controller.getClass(); cls != null; cls = cls.getSuperclass()) {
            if (SCAN_CANDIDATE_CLASS.equals(cls.getName())) {
                return cls;
            }
        }
        return null;
    }

    /**
     * Reads one field as declared on one exact class -- the resolution the
     * scan's own {@code getfield} performs, which a hierarchy walk from the
     * instance's class gets wrong here: the obfuscated subclasses redeclare
     * the same one-letter names with different types, and the walk returned
     * those shadows as if they were the candidate state (wiki log
     * 2026-09-11).
     *
     * @param owner The declaring class, from {@link #candidateClass}.
     * @param target The instance to read.
     * @param name The field, declared on {@code owner}.
     * @return The value.
     * @throws IllegalStateException When the field is absent or unreadable
     *     on the owner, which means the pinned layout moved.
     */
    private static Object readDeclared(Class<?> owner, Object target, String name) {
        try {
            java.lang.reflect.Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field.get(target);
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException(
                    "rw-agent: candidate class " + owner.getName()
                            + " no longer declares field " + name,
                    e);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read declared field " + name + " on " + owner.getName(),
                    e);
        }
    }

    /**
     * Reads one field reflectively, walking the class hierarchy, or null
     * when no class in it declares the name -- absence is a real answer on
     * a mixed roster.
     *
     * @param target The object to read.
     * @param name The field to look for, on the object's class or any
     *     superclass.
     * @return The value, or null when the field is absent everywhere.
     * @throws IllegalStateException When the field exists but cannot be
     *     read, which means the access machinery moved, not the layout.
     */
    private static Object readInherited(Object target, String name) {
        for (Class<?> cls = target.getClass(); cls != null; cls = cls.getSuperclass()) {
            java.lang.reflect.Field field;
            try {
                field = cls.getDeclaredField(name);
            } catch (NoSuchFieldException absent) {
                continue; // The hierarchy walk IS the search; try the parent.
            }
            try {
                field.setAccessible(true);
                return field.get(target);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(
                        "rw-agent: cannot read declared field " + name + " on "
                                + target.getClass().getName(),
                        e);
            }
        }
        return null;
    }

    private static String render(Object value) {
        return value == null ? "null" : value.toString();
    }
}
