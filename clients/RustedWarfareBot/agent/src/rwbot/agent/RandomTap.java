package rwbot.agent;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * Counts every draw from the engine's generator, by caller, between samples.
 *
 * <p>The RNG ledger proved the disease without naming the patient: two
 * identical pinned runs -- fast AND realtime alike -- held bit-identical
 * worlds for thousands of frames while the engine generator's states
 * desynced by frame 150. Something draws from that stream at a run-varying
 * rate whose values usually touch nothing; the world only forks when a
 * consequential consumer finally reads the shifted stream
 * (wiki: policy-determinism). A state mismatch says a stream desynced; only
 * a per-caller draw count says WHO drew differently.
 *
 * <p>So the generator is replaced outright: every {@code nextFloat},
 * {@code nextInt} and friend funnels through {@link Random#next}, so one
 * override observes every draw. Each draw is attributed to the nearest
 * engine-owned stack frame outside the draw-helper class itself and tallied;
 * {@link #flush} logs and clears the tally at each sample boundary. Two runs'
 * tap logs, diffed window by window, name the call site whose count varies --
 * the divergence, by class, method and line.
 *
 * <p>The swap needs {@code sun.misc.Unsafe}: the holder field is
 * {@code static final}, which reflection on this JVM (13) refuses to write.
 * Diagnostic only, armed by an explicit option, inert otherwise: walking a
 * stack per draw is not free, and the tap exists to be removed.
 */
final class RandomTap {

    private RandomTap() {
    }

    /** Package prefix a draw is attributed to; JDK and agent frames are skipped. */
    private static final String ENGINE_PREFIX = "com.corrodinggames.";

    /** The live tap, or null when never armed. */
    private static Tap tap;

    /** Whether {@link #install(long)} should do anything, set at premain. */
    private static boolean requested;

    /**
     * Records that the tap was asked for; the swap itself waits for the match.
     *
     * <p>The first build swapped the field at premain and measured nothing:
     * the holder class had not initialised yet, and its {@code <clinit>}
     * later overwrote the tap with the engine's own fresh generator -- a
     * silent failure that reported every window as identical because it was
     * counting a generator nothing used. So arming only registers intent, and
     * {@link #install(long)} runs at match start, long after initialisation,
     * and verifies the swap by reading the field back.
     */
    static void arm() {
        requested = true;
        Log.info("draw tap requested; installs at match start");
    }

    /** Whether the tap was asked for, deciding who gets the generator slot. */
    static boolean requested() {
        return requested;
    }

    /**
     * Replaces the engine's generator with the counting tap, if requested.
     *
     * <p>Called at match start, before the seeds are applied: the reseed
     * reads the holder field afresh and calls {@code setSeed} on whatever it
     * finds, so the tap inherits the pin with no special handling.
     *
     * @param seed The seed the tap starts from, matching what the pin will
     *     install anyway.
     * @throws IllegalStateException When Unsafe or the holder field cannot be
     *     reached, or the read-back does not return the tap -- a tap that
     *     silently failed to install would read as a clean run.
     */
    static void install(long seed) {
        if (!requested) {
            return;
        }
        Tap installed = new Tap(seed);
        EngineRandom.swapEngineGenerator(installed, "the draw tap");
        tap = installed;
        Log.info("draw tap installed on the engine generator");
    }

    /**
     * Logs and clears the per-caller tally for the window just sampled.
     *
     * <p>Runs on the game thread at each sample boundary, beside the ledger
     * line it exists to explain. A no-op when the tap was never armed. Sites
     * are emitted in name order so two logs diff cleanly.
     *
     * @param frame The boundary frame, for lining two logs up.
     */
    static void flush(int frame) {
        if (tap == null) {
            return;
        }
        StringBuilder line = new StringBuilder("rngtap frame=").append(frame);
        for (Map.Entry<String, Integer> site : new TreeMap<String, Integer>(tap.drain()).entrySet()) {
            line.append(' ').append(site.getKey()).append('=').append(site.getValue());
        }
        Log.info(line.toString());
    }

    /**
     * The counting generator: the tick-split with the sim-draw seam recording.
     *
     * <p>Extending the split rather than replacing it, deliberately: with
     * routing live, cosmetic draws cannot desync the simulation, so the only
     * draws worth diffing between runs are the ones {@code onSimDraw} sees.
     */
    private static final class Tap extends SplitRandom {

        private static final long serialVersionUID = 1L;

        /** Draws per attributed call site since the last drain. */
        private final Map<String, Integer> bySite = new HashMap<String, Integer>();

        Tap(long seed) {
            super(seed);
        }

        @Override
        void onSimDraw() {
            // bySite exists after the superclass constructor ran; a draw made
            // DURING super(seed) would see null, and Random(long) makes none.
            // Guarded anyway: a future JDK changing that must not turn the
            // tap into a boot crash.
            if (bySite != null) {
                record();
            }
        }

        /** Attributes one draw to its thread and nearest engine frame, and tallies it. */
        private void record() {
            StackTraceElement[] stack = Thread.currentThread().getStackTrace();
            String site = "unattributed";
            for (StackTraceElement element : stack) {
                String name = element.getClassName();
                // The helper class that owns the generator is skipped: every
                // draw passes through it, so it names nothing.
                if (name.startsWith(ENGINE_PREFIX)
                        && !name.equals(EngineNames.RANDOM_HOLDER_CLASS)) {
                    site = name.substring(ENGINE_PREFIX.length())
                            + "." + element.getMethodName() + ":" + element.getLineNumber();
                    break;
                }
            }
            // The thread is part of the name: a second thread sharing the
            // sim's stream is exactly the suspect this tap exists to catch
            // (boundary states that wobble and re-converge, wiki:
            // policy-determinism).
            site = Thread.currentThread().getName() + "@" + site;
            synchronized (bySite) {
                Integer sofar = bySite.get(site);
                bySite.put(site, sofar == null ? 1 : sofar.intValue() + 1);
            }
        }

        /** Returns the tally and starts a fresh one. */
        Map<String, Integer> drain() {
            synchronized (bySite) {
                Map<String, Integer> out = new HashMap<String, Integer>(bySite);
                bySite.clear();
                return out;
            }
        }
    }
}
