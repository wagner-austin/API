package rwbot.agent;

import java.lang.reflect.Method;

/**
 * Hosts a network game a human can join, and starts it when one does.
 *
 * <p>The sparring path (wiki: multiplayer-portability-invariants), split from
 * {@link MatchSetup} because the two share almost nothing beyond waiting for
 * the game thread: none of the reproducibility machinery runs here, and none
 * of the wrong-world guard either -- a human in the lobby can see what map
 * they are joining, and halting the JVM under them would tear down a session
 * a person is sitting in.
 *
 * <p>The whole chain lives in the engine's unobfuscated script surface — the
 * same one that provided {@code -sandbox}: {@code hostStart(false)} boots a
 * private LAN server on the configured port and opens the battleroom;
 * {@code mp.setMapFromPopup(path)} broadcasts the map; and the battleroom's
 * own Start button reduces to {@code mp.multiplayerStart()}, whose server
 * branch has no readiness gate. The lobby is polled through the same player
 * roster the state stream already reads, so "a human joined" is "a second
 * non-absent slot".
 *
 * <p><b>None of the reproducibility machinery runs here, by design.</b> No
 * hold — a human cannot be world-held; no reseed, no frame zeroing, no fixed
 * logic step — the peers simulate in lockstep and rewriting either side's
 * clock is a desync. The planner must be free-running
 * ({@code lockstepFrames=0}); this is invariant four of the portability page,
 * selected rather than assumed.
 */
final class HostSetup {

    private HostSetup() {
    }

    /**
     * Schedules a hosted network game on a daemon thread.
     *
     * @param map Map path as the engine names it, e.g.
     *     {@code maps/skirmish/[p2]Lake (2p).tmx}.
     * @param channel The command channel to open once the match is live.
     */
    static void schedule(String map, CommandChannel channel) {
        Thread thread = new Thread(() -> run(map, channel), "rw-agent-host");
        thread.setDaemon(true);
        thread.start();
        Log.info("hosting requested on " + map + "; a human may join");
    }

    /** Boots the lobby, waits for a human, starts the game. Daemon thread. */
    private static void run(String map, CommandChannel channel) {
        if (!MatchSetup.awaitGameThread()) {
            Log.error("hosting abandoned: the game thread never became ready");
            return;
        }
        if (!MatchSetup.settle()) {
            Log.error("hosting interrupted while settling");
            return;
        }
        // hostStart(false) is "Host Private" — the battleroom's own popup
        // wires exactly this call. The map is broadcast separately because
        // hostStart resets the selection to the engine's eight-player
        // default whenever none is set.
        queueScript("hostStart(false);");
        queueScript("mp.setMapFromPopup('" + map + "');");
        Log.info("lobby open; waiting for a player to join before starting");
        int waited = 0;
        while (Scoreboard.rosterCount() < 2) {
            try {
                Thread.sleep(1000L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Log.error("hosting interrupted while waiting for a player");
                return;
            }
            waited++;
            if (waited % 15 == 0) {
                Log.info("still waiting for a player to join (" + waited + "s)");
            }
        }
        Log.info("player joined; starting the game");
        queueScript("mp.multiplayerStart();");
        Orders.onGameThread(() -> watchForHostedMatch(channel));
    }

    /**
     * Runs each tick until the hosted match is live, then opens the channel.
     *
     * <p>The same liveness predicate the skirmish watcher trusts — the local
     * player exists and owns its starting units — and nothing else: the
     * world is not held, reseeded, re-clocked or re-paced, because every one
     * of those is a desync against the human's client.
     */
    private static void watchForHostedMatch(CommandChannel channel) {
        Object engine = EngineHandle.current();
        boolean live =
                engine != null
                        && EngineAccess.readField(engine, EngineNames.LOCAL_TEAM) != null
                        && !Perception.ownedUnits(engine).isEmpty();
        if (!live) {
            Orders.onGameThread(() -> watchForHostedMatch(channel));
            return;
        }
        channel.start();
        Log.info("hosted match live; channel open, world running free for the human");
    }

    /** Queues one script string on the engine's own script engine. */
    private static void queueScript(String script) {
        Orders.onGameThread(
                () -> {
                    Class<?> scripts = EngineAccess.pinnedClass(EngineNames.SCRIPTS_CLASS);
                    Object instance = EngineAccess.invokeStatic(scripts, "getInstance");
                    Method queue =
                            EngineAccess.pinnedMethod(
                                    scripts, EngineNames.SCRIPT_QUEUE_METHOD, String.class);
                    EngineAccess.invoke(queue, instance, script);
                });
    }
}
