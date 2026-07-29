package rwbot.agent;

import java.lang.reflect.Method;

/**
 * Starts the match the experiment asked for, instead of the one the engine
 * hardcodes.
 *
 * <p><b>Nobody chose what the bot has been playing.</b> {@code -sandbox} queues
 * a fixed script naming a ten-player map, and the setup it loads is read out of
 * a GUI document that headless has no values in — so every figure falls through
 * to a Java default:
 *
 * <pre>
 *   aiDifficulty = getElementById("aiDifficulty").getValueAsInt(0);   // Medium
 *   numberOfAIs  = getElementById("numberOfAIs").getValueAsInt(4);    // four opponents
 *   aiTeams      = getElementById("aiTeams").getValueAsInt(1);
 * </pre>
 *
 * Four opponents, none of which is ever eliminated, on <i>Crossing Large
 * (10p)</i>. Editing the document does not help: it was tried, and a match
 * still reported five players, because the element has no value to read rather
 * than the wrong one (wiki: policy-determinism).
 *
 * <p>So this bypasses the GUI and calls what the GUI would have called. The
 * count of enemies is {@code opponents - allies} and is capped by the map's own
 * team count, so a two-player map is a duel whatever is asked for.
 *
 * <p>Difficulty is set immediately before, because the same path would
 * otherwise overwrite it: {@code loadConfigCommon} assigns the settings field
 * from the GUI fallback and saves, which is why {@code preferences.ini} reads
 * {@code aiDifficulty:0} however it is edited. The scale is -2 Very Easy, -1
 * Easy, 0 Medium, 1 Hard, 2 Very Hard, 3 Impossible, and it is an <b>income
 * multiplier on the AI alone</b> — 0.4x, 0.7x, 1.0x, 1.4x, 1.8x and 3.7x. At
 * Medium an opponent earns exactly what the bot does.
 */
final class MatchSetup {

    /** Seconds to let the engine settle after the game thread first answers. */
    private static final int SETTLE_SECONDS = 4;

    /** How long to wait for the game thread before giving up. */
    private static final int READY_TIMEOUT_SECONDS = 90;

    private MatchSetup() {
    }

    /**
     * Schedules the match on a daemon thread.
     *
     * <p>Daemon so a failed setup can never hold the JVM open, and off the game
     * thread because the wait is a poll.
     *
     * @param map Map path as the engine names it, e.g.
     *     {@code maps/skirmish/[p2]Lake (2p).tmx}.
     * @param opponents How many AI players to face.
     * @param difficulty AI difficulty, -2 to 3.
     */
    static void schedule(String map, int opponents, int difficulty, Runnable afterStarted) {
        Thread thread =
                new Thread(() -> run(map, opponents, difficulty, afterStarted), "rw-agent-match");
        thread.setDaemon(true);
        thread.start();
        Log.info(
                "match requested: "
                        + opponents
                        + " opponent(s) at difficulty "
                        + difficulty
                        + " on "
                        + map);
    }

    /** Waits for the engine, then starts the requested match on the game thread. */
    private static void run(String map, int opponents, int difficulty, Runnable afterStarted) {
        if (!awaitGameThread()) {
            Log.error(
                    "match setup abandoned: the game thread never became ready within "
                            + READY_TIMEOUT_SECONDS
                            + "s");
            return;
        }
        // The engine queues its own sandbox load at startup. Ours replaces it,
        // so it has to arrive after -- and before the planner attaches, which
        // it does well inside the settle the recipe already waits out.
        try {
            Thread.sleep(SETTLE_SECONDS * 1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("match setup interrupted while settling");
            return;
        }
        java.util.concurrent.CountDownLatch started =
                new java.util.concurrent.CountDownLatch(1);
        Orders.onGameThread(
                () -> {
                    try {
                        start(map, opponents, difficulty);
                    } finally {
                        started.countDown();
                    }
                });
        // **Nothing may sample until the match exists.** Starting a match
        // replaces the engine's game object, so a channel armed beforehand
        // counts frames against a game that is thrown away: the planner
        // connected, the lockstep hook waited on the old simulation, and the
        // first sample never arrived. The recipe waits for the port, so
        // opening it late costs a few seconds and nothing else.
        try {
            if (!started.await(READY_TIMEOUT_SECONDS, java.util.concurrent.TimeUnit.SECONDS)) {
                Log.error("match setup abandoned: the game thread never ran the start");
                return;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("match setup interrupted awaiting the start");
            return;
        }
        // **Difficulty is set last, because the load overwrites it.**
        // `loadConfigCommon` assigns the settings field from the GUI's unread
        // default and saves, which is why preferences.ini reads
        // `aiDifficulty:0` however it is edited. The AI re-reads the field as
        // it runs, so setting it after the match exists is what sticks.
        try {
            Thread.sleep(SETTLE_SECONDS * 1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("match setup interrupted before setting difficulty");
            return;
        }
        Orders.onGameThread(() -> applyDifficulty(difficulty));
        afterStarted.run();
    }

    /** Sets the AI difficulty on the live match. Runs on the game thread. */
    private static void applyDifficulty(int difficulty) {
        Object engine = EngineHandle.current();
        Object settings = EngineAccess.readField(engine, EngineNames.SETTINGS_FIELD);
        java.lang.reflect.Field field =
                EngineAccess.pinnedField(settings.getClass(), EngineNames.AI_DIFFICULTY_FIELD);
        try {
            field.setInt(settings, difficulty);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot set " + EngineNames.AI_DIFFICULTY_FIELD, e);
        }
        Log.info("difficulty set to " + difficulty);
    }

    /**
     * Polls until work can be queued on the game thread.
     *
     * @return True when the game thread answered, false on timeout.
     */
    private static boolean awaitGameThread() {
        long deadline = System.nanoTime() + READY_TIMEOUT_SECONDS * 1_000_000_000L;
        while (System.nanoTime() < deadline) {
            if (Orders.gameThreadReady()) {
                return true;
            }
            try {
                Thread.sleep(250L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return false;
    }

    /**
     * Sets the difficulty and starts the match. Runs on the game thread.
     *
     * @throws IllegalStateException When a pinned name is absent, which means
     *     the obfuscated layout moved and the match would otherwise start with
     *     silently different settings.
     */
    private static void start(String map, int opponents, int difficulty) {
        // **The engine's own path, with our map substituted, and nothing else
        // reimplemented.** Calling the match-setup helper directly was tried
        // and is a trap: it loads the terrain but leaves the simulation
        // stopped, because the caller does the starting -- and the caller is a
        // chain of GUI steps. Every attempt to replay that chain by hand
        // reached a different half-started state.
        //
        // **The opponent count needs no override.** The helper caps teams by
        // the map's own count, so a two-player map yields exactly one enemy
        // whatever the GUI's unread default says. Choosing the map is choosing
        // the opponent count, which is why `matchOpponents` is reported here
        // rather than sent.
        Class<?> scripts = EngineAccess.pinnedClass(EngineNames.SCRIPTS_CLASS);
        Object instance = EngineAccess.invokeStatic(scripts, "getInstance");
        Method queue =
                EngineAccess.pinnedMethod(
                        scripts, EngineNames.SCRIPT_QUEUE_METHOD, String.class);
        String script =
                "open('sandboxOptions.rml', '" + map + "'); loadConfigAndStartNewSandbox('" + map
                        + "');";
        EngineAccess.invoke(queue, instance, script);
        Log.info("match starting on " + map + "; expecting " + opponents + " opponent(s)");
    }
}
