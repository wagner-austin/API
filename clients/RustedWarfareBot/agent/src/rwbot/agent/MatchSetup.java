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

    /**
     * Fixed logic step, in milliseconds, for a reproducible match.
     *
     * <p>The simulation advances by the wall-measured frame delta -- the
     * container stores it, scales it by 0.06 and hands it to the game -- so
     * CPU load is part of the physics: twelve certification replicas of one
     * seed played in parallel produced twelve distinct scorecards while
     * sequential pairs of the same specification agreed to the credit
     * (wiki: policy-determinism). Slick's own fixed-timestep machinery closes
     * this: with the container's minimum and maximum logic intervals set
     * equal, the accumulate-and-chunk loop hands the game that exact figure
     * on every update -- the remainder is carried, never delivered -- and the
     * measured delta stops existing below the accumulator. Three milliseconds
     * is the engine's own 300 fps target rounded to the integer the setters
     * take.
     */
    private static final int FIXED_LOGIC_MS = 3;

    /**
     * The engine's built-in fixed-delta override: {@code gameFramework.l.bu},
     * read as {@code if (bu >= 0) f2 = bu} at the top of every tick and
     * {@code -1} (off) by default. The one field that pins the simulation's
     * step size inside the engine itself, wherever the delta came from.
     * Pinned to 1.15 (code 176, build #28) like every obfuscated name.
     */
    private static final String DELTA_OVERRIDE_FIELD = "bu";

    /**
     * Parks the ambient spawner's accumulator so far below its 10.0 firing
     * threshold that no run length can climb back: ~1e31 ticks at the pinned
     * delta, against 3e4 in the longest match.
     */
    private static final float AMBIENT_NEVER = -1.0e30f;

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
     * @param seed Engine random seed to re-apply the moment the match exists,
     *     or zero to leave the generators as they are.
     * @param channel The command channel to open and arm once the match
     *     exists, so nothing samples the game object that is thrown away.
     * @param pinDeltaMs Constant frame delta to pin the simulation to, in
     *     milliseconds, or zero to leave the engine on the wall clock. Zero is
     *     what a spectator run passes: a pinned clock decouples simulation
     *     speed from wall time, which is right for the harness and wrong for
     *     a human watching.
     */
    static void schedule(
            String map,
            int opponents,
            int difficulty,
            long seed,
            CommandChannel channel,
            int pinDeltaMs,
            int fastForwardFps) {
        Thread thread =
                new Thread(
                        () ->
                                run(
                                        map,
                                        opponents,
                                        difficulty,
                                        seed,
                                        channel,
                                        pinDeltaMs,
                                        fastForwardFps),
                        "rw-agent-match");
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
    private static void run(
            String map,
            int opponents,
            int difficulty,
            long seed,
            CommandChannel channel,
            int pinDeltaMs,
            int fastForwardFps) {
        if (!awaitGameThread()) {
            Log.error(
                    "match setup abandoned: the game thread never became ready within "
                            + READY_TIMEOUT_SECONDS
                            + "s");
            return;
        }
        // The engine queues its own sandbox load at startup. Ours replaces it,
        // so it has to arrive after. Wall clock is tolerable here and only
        // here: everything before the match starts happens to a game object
        // that starting the match throws away.
        try {
            Thread.sleep(SETTLE_SECONDS * 1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("match setup interrupted while settling");
            return;
        }
        // From here on, every step runs on the game thread and is measured in
        // ticks rather than seconds. The start runnable reseeds -- so the
        // load's own draws, unit spawn scatter among them, come from a known
        // generator state -- queues the match script, and posts a watcher
        // that holds the world on the first tick the match is live.
        //
        // **Why liveness and not earlier.** Holding on this same tick and
        // metering the load itself through lockstep was tried, and two runs
        // diverged in content: the load does wall-clock work off the game
        // thread, so pacing the game thread through it changes what
        // interleaves with what. Held at liveness instead, two runs produce
        // identical content at identical match age, one frame of phase
        // apart -- the load runs exactly as it always did, and the hold
        // begins the moment its result exists (wiki: policy-determinism).
        Orders.onGameThread(
                () -> {
                    if (seed != 0) {
                        EngineRandom.seed(seed);
                    }
                    start(map, opponents, difficulty);
                    // The watcher rides the PRE-tick queue from here on. The
                    // script queue drains AFTER the simulation ticks, so a
                    // drain-time watcher could not latch until the new world
                    // had already run free ticks -- wall-clock-valued ones,
                    // whose pollution of the opponents' think-timer floats
                    // was the last measured divergence between pinned runs
                    // (wiki: policy-determinism). The pre-tick queue drains
                    // at the top of the tick body, before the world updates,
                    // so the watcher latches BEFORE the new world's first
                    // update. The engine object itself is a stable singleton
                    // -- starting a match replaces its game state, not the
                    // instance -- so the queue survives the start.
                    Orders.onEngineTick(
                            EngineHandle.current(),
                            () ->
                                    watchForMatch(
                                            difficulty, seed, channel, pinDeltaMs, fastForwardFps));
                });
    }

    /**
     * Runs each tick until the match is live, then finishes the setup and
     * holds the world on that same tick.
     *
     * <p><b>Liveness is engine state, not object identity.</b> The engine
     * singleton is constructed exactly once -- {@code l.a(Context, n)} returns
     * the existing instance forever after -- so "the game object was replaced"
     * is not observable and was never the right test. What a started match
     * does change is the state on that singleton: the local player exists and
     * owns its starting units. Both are read through the same pinned names the
     * order path already trusts.
     *
     * <p><b>Difficulty is set here, because the load overwrites it.</b>
     * {@code loadConfigCommon} assigns the settings field from the GUI's
     * unread default and saves, which is why preferences.ini reads
     * {@code aiDifficulty:0} however it is edited. The AI re-reads the field
     * as it runs, so setting it on the match's first live tick both sticks
     * and lands at the same match age every run.
     *
     * <p><b>The seed is re-applied here, and this is not belt-and-braces.</b>
     * Both pinned generators are global and the load consumes draws from
     * them. Seeding before the script pins the load itself; reseeding now
     * hands the match a generator state that is exactly the seed again,
     * whatever the load consumed. Together they make "same seed" mean "same
     * match".
     */
    private static void watchForMatch(
            int difficulty,
            long seed,
            CommandChannel channel,
            int pinDeltaMs,
            int fastForwardFps) {
        Object engine = EngineHandle.current();
        boolean live =
                engine != null
                        && EngineAccess.readField(engine, EngineNames.LOCAL_TEAM) != null
                        && !Perception.ownedUnits(engine).isEmpty();
        if (!live) {
            // Re-posted to the PRE-tick queue: liveness flips on the loader
            // thread mid-pass, and this check must run before the next
            // tick's world update, not after it -- the whole reason the
            // watcher rides this queue (see awaitEngineSwap).
            Orders.onEngineTick(
                    engine,
                    () -> watchForMatch(difficulty, seed, channel, pinDeltaMs, fastForwardFps));
            return;
        }
        // The generator swap happens before the reseed reads the field, so
        // the seed lands on the replacement. The tap (diagnostic, measures
        // the RAW shared stream) takes the slot when asked for; otherwise a
        // seeded match gets the thread-split generator, which is what makes
        // the simulation's draws a pure function of the seed while the
        // particle thread draws its own side stream (see SplitRandom).
        if (RandomTap.requested()) {
            RandomTap.install(seed);
        } else if (seed != 0) {
            SplitRandom.install(seed);
        }
        if (seed != 0) {
            EngineRandom.seed(seed);
        }
        applyDifficulty(difficulty);
        pinLogicInterval(fastForwardFps);
        // **The engine's own delta override, and the end of the wall clock.**
        // Pinning the container's logic interval fixes what each update()
        // carries, but not how many updates precede a render -- a render-only
        // cycle ticks the simulation with the zeroed leftover delta, so which
        // ticks were zero-length still followed OS scheduling. The engine
        // ships the kill switch itself: `if (bu >= 0) f2 = bu` sits inside
        // the tick, downstream of every container quirk, unused at its -1
        // default. Set, it makes every tick the same fixed quantum of
        // simulation whatever the container measured -- proven bit-identical
        // over 600 full-planner samples where every weaker pin still wobbled
        // (wiki: policy-determinism, engine-tick-and-clock).
        if (pinDeltaMs > 0) {
            EngineAccess.writeFloatField(engine, DELTA_OVERRIDE_FIELD, pinDeltaMs * 0.06f);
        }
        // Fast-forward drives the engine's own tick entry, so it arms on
        // the engine rather than the container (see FastForward for the
        // eight measured reasons the container was the wrong tree).
        FastForward.arm(engine, fastForwardFps, pinDeltaMs > 0 ? pinDeltaMs : FIXED_LOGIC_MS);
        // **The ambient spawner is silenced on seeded matches.** Its
        // accumulator counts the MEASURED delta, so every ~10 units of wall
        // time it spends two draws from the sim's shared random stream on a
        // cosmetic effect at a random map tile -- the draw tap measured the
        // counts wobbling 1-vs-2 per window between identical pinned runs,
        // and every downstream AI roll inherited the shift. Parked far below
        // its 10.0 threshold it can never fire, and f.a(0,0) aside, no other
        // wall-paced drawer remains (wiki: policy-determinism). Cosmetic
        // only: watch and host runs, unseeded, keep their ambience.
        if (seed != 0) {
            Object effects = EngineAccess.readField(engine, EngineNames.EFFECTS_MANAGER);
            EngineAccess.writeFloatField(effects, EngineNames.AMBIENT_CLOCK, AMBIENT_NEVER);
            Log.info("ambient spawner silenced; its wall-paced draws are off the sim stream");
        }
        // **The frame counter is part of the randomness, so it is reset.**
        // The engine's synced draws hash the match seed with the frame
        // counter -- `f.a(min,max,salt)` mixes `l2.bx` into the result four
        // ways -- and `bx` counts from boot, so two runs whose menus lasted a
        // different number of frames drew differently on every call however
        // thoroughly the generators were seeded. Zeroing frame and clock here
        // is the engine's own new-game convention (`j.ad.aD()` stores literal
        // zero into both) performed at the moment this start path skips it
        // (wiki: policy-determinism, engine-tick-and-clock).
        EngineAccess.writeIntField(engine, StateStream.FRAME_FIELD, 0);
        EngineAccess.writeIntField(engine, StateStream.CLOCK_FIELD, 0);
        // The AI's think-timers are wall-polluted by the load's free ticks
        // exactly like the counters above; they go back to constructed state
        // the same moment (see AiTimers).
        if (seed != 0) {
            AiTimers.reset();
        }
        channel.start();
        // Held synchronously, on this very tick: posting the hold instead
        // would let the world run free for however many ticks the queue
        // takes, and those are exactly the frames this design exists to
        // remove.
        channel.holdNow();
        Log.info("match live; frame zeroed, channel open, world held for the planner");
    }

    /**
     * Schedules a hosted network game a human can join, on a daemon thread.
     *
     * <p>The sparring path (wiki: multiplayer-portability-invariants). The
     * whole chain lives in the engine's unobfuscated script surface — the
     * same one that provided {@code -sandbox}: {@code hostStart(false)}
     * boots a private LAN server on the configured port and opens the
     * battleroom; {@code mp.setMapFromPopup(path)} broadcasts the map; and
     * the battleroom's own Start button reduces to {@code
     * mp.multiplayerStart()}, whose server branch has no readiness gate.
     * The lobby is polled through the same player roster the state stream
     * already reads, so "a human joined" is "a second non-absent slot".
     *
     * <p><b>None of the reproducibility machinery runs here, by design.</b>
     * No hold — a human cannot be world-held; no reseed, no frame zeroing,
     * no fixed logic step — the peers simulate in lockstep and rewriting
     * either side's clock is a desync. The planner must be free-running
     * ({@code lockstepFrames=0}); this is invariant four of the portability
     * page, selected rather than assumed.
     *
     * @param map Map path as the engine names it, e.g.
     *     {@code maps/skirmish/[p2]Lake (2p).tmx}.
     * @param channel The command channel to open once the match is live.
     */
    static void scheduleHost(String map, CommandChannel channel) {
        Thread thread = new Thread(() -> runHost(map, channel), "rw-agent-host");
        thread.setDaemon(true);
        thread.start();
        Log.info("hosting requested on " + map + "; a human may join");
    }

    /** Boots the lobby, waits for a human, starts the game. Daemon thread. */
    private static void runHost(String map, CommandChannel channel) {
        if (!awaitGameThread()) {
            Log.error(
                    "hosting abandoned: the game thread never became ready within "
                            + READY_TIMEOUT_SECONDS
                            + "s");
            return;
        }
        try {
            Thread.sleep(SETTLE_SECONDS * 1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
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

    /**
     * Pins the container to a fixed logic step, so CPU load leaves the
     * simulation.
     *
     * <p>The container is {@code Main.m.k}, an {@code AppGameContainer}
     * subclass, and the setters are Slick's own public, unobfuscated API --
     * the one reflective walk here that cannot be broken by a game update
     * re-obfuscating names. Failing loudly on the two package fields keeps
     * the pin philosophy: a run that silently kept wall-clock physics would
     * read as reproducible and not be.
     *
     * @throws IllegalStateException When the route to the container is
     *     absent, which means the pinned layout moved.
     */
    /**
     * Reads the game container off {@code Main.m.k}, or null before it is
     * built. Package-visible so the fast-forward diagnostic can compare the
     * instance it spoofed against the one currently installed.
     */
    static Object liveContainer() {
        try {
            Class<?> mainClass =
                    Class.forName(
                            "com.corrodinggames.rts.java.Main",
                            false,
                            Orders.class.getClassLoader());
            java.lang.reflect.Field instance = mainClass.getDeclaredField("m");
            instance.setAccessible(true);
            java.lang.reflect.Field held = mainClass.getDeclaredField("k");
            held.setAccessible(true);
            return held.get(instance.get(null));
        } catch (ClassNotFoundException | NoSuchFieldException | IllegalAccessException e) {
            return null;
        }
    }

    private static void pinLogicInterval(int fastForwardFps) {
        Object container;
        try {
            Class<?> mainClass =
                    Class.forName(
                            "com.corrodinggames.rts.java.Main",
                            false,
                            MatchSetup.class.getClassLoader());
            java.lang.reflect.Field instance = mainClass.getDeclaredField("m");
            instance.setAccessible(true);
            java.lang.reflect.Field held = mainClass.getDeclaredField("k");
            held.setAccessible(true);
            container = held.get(instance.get(null));
        } catch (ClassNotFoundException | NoSuchFieldException | IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot reach the game container through Main.m.k"
                            + EngineNames.PIN,
                    e);
        }
        if (container == null) {
            throw new IllegalStateException(
                    "rw-agent: Main.m.k is null; the container has not been built yet");
        }
        Method minimum =
                EngineAccess.pinnedMethod(
                        container.getClass(), "setMinimumLogicUpdateInterval", int.class);
        Method maximum =
                EngineAccess.pinnedMethod(
                        container.getClass(), "setMaximumLogicUpdateInterval", int.class);
        EngineAccess.invoke(minimum, container, Integer.valueOf(FIXED_LOGIC_MS));
        EngineAccess.invoke(maximum, container, Integer.valueOf(FIXED_LOGIC_MS));
        Log.info(
                "logic step pinned to " + FIXED_LOGIC_MS
                        + "ms; the simulation no longer measures the wall clock");

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
        // **The opponent count IS sent, and the belief that it need not be
        // cost a whole measurement arc.** The load path reads
        // {@code numberOfAIs} off the open document and falls through to the
        // Java default of FOUR when the element carries no value -- and the
        // map caps that at its spawn count, not at the number its name
        // advertises. Every "(2p)"-named map in the skirmish roster except
        // duel_lake seats four, so the whole cross-map arc silently played
        // 1v3 (wiki log 2026-08-05). {@code setValueById} is the engine's
        // own script-callable setter for exactly the attribute
        // {@code getValueAsInt} reads; writing it between the open and the
        // load is what the GUI itself would have done. Editing the .rml on
        // disk was tried long ago and does nothing -- the element has no
        // value ATTRIBUTE to read from the file; the live document is where
        // the value lives.
        Class<?> scripts = EngineAccess.pinnedClass(EngineNames.SCRIPTS_CLASS);
        Object instance = EngineAccess.invokeStatic(scripts, "getInstance");
        Method queue =
                EngineAccess.pinnedMethod(
                        scripts, EngineNames.SCRIPT_QUEUE_METHOD, String.class);
        String script =
                "open('sandboxOptions.rml', '" + map + "'); "
                        + "setValueById('numberOfAIs', '" + opponents + "'); "
                        + "loadConfigAndStartNewSandbox('" + map + "');";
        EngineAccess.invoke(queue, instance, script);
        Log.info("match starting on " + map + "; asking for " + opponents + " opponent(s)");
    }
}
