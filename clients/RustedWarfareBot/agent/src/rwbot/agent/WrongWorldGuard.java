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
     * Seconds a requested match gets to go live before the run is killed.
     *
     * <p>Measured from the moment the start is queued, after the ready wait
     * and the settle. A duel map loads in single-digit seconds on a machine
     * running a full parallel sweep, so a minute is generous -- and it must
     * stay under the harness's own 90s port wait, because the port only
     * opens at liveness and whichever timeout fires first writes the
     * diagnosis: the agent's names the world that was live, the harness's
     * only says the port never opened.
     */
    private static final int LIVE_DEADLINE_SECONDS = 60;

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
        long deadline = System.nanoTime() + LIVE_DEADLINE_SECONDS * 1_000_000_000L;
        while (System.nanoTime() < deadline) {
            if (matchLive) {
                return;
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
                "requested " + map + " but after " + LIVE_DEADLINE_SECONDS + "s " + state
                        + " -- halting before a scorecard can be farmed from it");
        Runtime.getRuntime().halt(WRONG_WORLD_EXIT);
    }
}
