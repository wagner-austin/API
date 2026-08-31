package rwbot.agent;

/**
 * Kills a run whose requested match cannot happen, before it can be measured.
 *
 * <p>The failure this guards against was found in the wild wearing three
 * disguises. A configured match whose map file the game dir lacks fails its
 * load with an engine alert nothing headless reads, and the run drifts into
 * the boot sandbox instead — a world with a local player that owns units, so
 * liveness latches, the channel opens, and the harness collects a plausible
 * scorecard from a match nobody asked for. Six stale worker clones did
 * exactly that and voided the whole cross-map batch family
 * (wiki: policy-determinism, the seating anomaly). The clone-side fix is the
 * harness re-syncing maps on every prepare; this is the agent-side fix, so
 * the next cause of a wrong world — whatever it is — dies loudly too.
 *
 * <p>Three pieces, each catching a disguise the others cannot:
 *
 * <ul>
 *   <li>{@link #armMapLoadCrash} arms the engine's own testing switch, so a
 *       failed map load crashes at its origin instead of alerting nobody.
 *   <li>{@link #isRequestedWorld} gates the match watcher's latch on the
 *       engine stating the requested map, so no other world -- however
 *       complete -- can receive the setup or open the channel.
 *   <li>{@link #awaitDeadline} kills a run whose requested world never
 *       arrives, naming what was live instead, rather than idling until the
 *       harness loses patience with a silence it cannot attribute.
 * </ul>
 *
 * <p>Skirmish only. None of this runs on the hosted sparring path
 * ({@link HostSetup}): a human in the lobby can see what map they are
 * joining, and halting the JVM under them would tear down a session a person
 * is sitting in.
 */
final class WrongWorldGuard {

    /**
     * Seconds a requested match gets to go live before the run is killed,
     * while the engine has NOT adopted the requested map.
     *
     * <p>Measured from the moment the start is queued, after the ready wait
     * and the settle. A duel map loads in single-digit seconds on a machine
     * running a full parallel sweep, so a minute is generous for the case
     * this guard exists for: a request that evaporated, leaving some OTHER
     * world live. The old constraint that this must stay under the
     * harness's port wait dissolved when that wait became silence-keyed
     * (wait_for_channel): a loading engine keeps writing and keeps both
     * alive, and a dead one fails the harness side fast.
     */
    private static final int LIVE_DEADLINE_SECONDS = 60;

    /**
     * Seconds the load gets once the engine has ADOPTED the requested map.
     *
     * <p>A separate, longer deadline, because adoption excludes the disguise
     * class the sixty seconds exist for -- the map-path field carrying the
     * requested string means THIS match is loading, just slowly. Cluster
     * nodes under a 22-way submission burst measured a 56 second single
     * asset read against 4ms on quiet members, and members died at the flat
     * minute while demonstrably mid-load (jobs 55671486/55671507, wiki log
     * 2026-08-31). Five minutes covers that class with the same headroom
     * the minute gives a quiet load; a load that CRASHED goes silent and is
     * the launcher's silence budget to catch, not this deadline's.
     */
    private static final int ADOPTED_DEADLINE_SECONDS = 300;

    /**
     * Exit status when the run dies wrong-world or never-live -- sysexits'
     * EX_SOFTWARE, distinct from the self-test's 1 and 2.
     */
    private static final int WRONG_WORLD_EXIT = 70;

    /** Flipped by the match watcher on the tick the requested match goes live. */
    private static volatile boolean matchLive;

    private WrongWorldGuard() {
    }

    /**
     * Arms the engine's own automated-testing switch before the load.
     *
     * <p>The engine ships this half of the guard itself: with the flag set,
     * the map-load catch throws -- logging "Crashing on allowed map error
     * because automated testing is active" -- so a missing map is a crash at
     * its origin rather than an alert at nobody. Cleared only by the class's
     * static initializer, so the write sticks for the process. The flag's
     * one other reader appends a debug marker to the multiplayer handshake,
     * which does not matter to a skirmish and is why arming it here is free.
     */
    static void armMapLoadCrash() {
        EngineAccess.writeStaticBooleanField(
                EngineAccess.pinnedClass(EngineNames.ENGINE_CLASS),
                EngineNames.TESTING_FLAG,
                true);
        Log.info("map-load errors armed to crash; a missing map can no longer drift");
    }

    /**
     * Whether the engine's current world is the requested one.
     *
     * <p>The identity half, and it gates the latch rather than checking
     * after it. Liveness alone cannot tell worlds apart: the menu
     * background is a running mission demo whose local player owns units
     * whenever its script phase says so, and the start script executes a
     * full frame after the runnable that queues it -- the engine's script
     * drain snapshots its queue before running it, so a script queued by a
     * running action always waits for the next drain. Checked at the latch,
     * that one-frame window is a coin flip against the demo's phase; folded
     * into the liveness predicate, no world but the requested one can
     * receive the setup or open the channel, however the timing falls.
     *
     * <p>What the live world cannot hide is its map: every load path
     * writes the engine's map-path field before loading, and the match
     * starter stores the exact string it was handed -- inside the same
     * script-drain call that tears the old world down, so no tick boundary
     * can observe the new path over the old world.
     */
    static boolean isRequestedWorld(String map, Object engine) {
        return map.equals(EngineAccess.readField(engine, EngineNames.MAP_PATH));
    }

    /** Called by the match watcher on the tick the requested match goes live. */
    static void markLive() {
        matchLive = true;
    }

    /**
     * Holds the setup thread until the match goes live, or kills the JVM.
     *
     * <p>The deadline half, and with the latch gated on the map it is what
     * turns every wrong-world disguise into the same loud death. A missing
     * map's request evaporates inside the engine's GUI layer -- no alert,
     * no throw, the menu demo just keeps running -- and a load that fails
     * later crashes on the armed switch; either way the requested world
     * never arrives, the watcher never latches, and this deadline expires
     * naming what was live instead. Halt rather than exit: a run this
     * wrong has nothing worth flushing, and the harness reading a dead
     * process with no scorecard is the point. The map-path read races the
     * game thread, which is tolerable in a message printed by a run that
     * is already dead.
     */
    static void awaitDeadline(String map) {
        long start = System.nanoTime();
        // The deadline is re-derived every pass rather than fixed up front,
        // because which one applies is a fact that ARRIVES: the engine
        // adopts the requested map partway through the wait, and from that
        // moment the run is a slow load rather than a candidate disguise.
        int allowed = LIVE_DEADLINE_SECONDS;
        while (System.nanoTime() - start < allowed * 1_000_000_000L) {
            if (matchLive) {
                return;
            }
            Object polled = EngineHandle.current();
            if (allowed == LIVE_DEADLINE_SECONDS
                    && polled != null
                    && isRequestedWorld(map, polled)) {
                allowed = ADOPTED_DEADLINE_SECONDS;
                Log.info(
                        "the engine adopted " + map + "; the load gets "
                                + ADOPTED_DEADLINE_SECONDS + "s to go live");
            }
            try {
                Thread.sleep(500L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
        Object engine = EngineHandle.current();
        Object live = engine == null ? null : EngineAccess.readField(engine, EngineNames.MAP_PATH);
        String state =
                map.equals(live)
                        ? "the load started but never produced a live world -- its crash is"
                                + " earlier in this log"
                        : "the live world is " + live;
        Log.error(
                "requested " + map + " but after " + allowed + "s " + state
                        + " -- halting before a scorecard can be farmed from it");
        Runtime.getRuntime().halt(WRONG_WORLD_EXIT);
    }
}
