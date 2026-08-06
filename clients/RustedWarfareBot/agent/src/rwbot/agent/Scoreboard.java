package rwbot.agent;

/**
 * What the players are worth, as the engine's own scoreboard keeps it.
 *
 * <p>Reads only, like {@link Perception}, which is where this was split from
 * when that class reached 943 lines. The seam is the subject: everything here
 * answers a question about a <i>player</i> -- credits, defeat, income, army and
 * building value, how many are left -- where perception answers questions about
 * entities on the map.
 *
 * <p>Every figure is read from the engine's statistics rather than counted from
 * the roster. It charts all three values per player and writes them into its
 * own save file, so these are the numbers the game itself would show, and
 * anything recomputed here could disagree with them (wiki:
 * perception-visibility).
 */
final class Scoreboard {

    private Scoreboard() {
    }

    /**
     * Returns the current player's credits, rounded down to whole currency.
     *
     * <p>The engine holds this as a double and spends it in whole units, so a
     * planner comparing against a unit price wants the floor rather than the
     * raw value: 99.97 credits does not buy a 100-credit structure.
     *
     * @param engine The live engine instance.
     * @return Credits, or 0 when there is no current player.
     */
    static int creditsOf(Object engine) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return 0;
        }
        return (int) Math.floor(EngineAccess.readDoubleField(team, EngineNames.CREDITS));
    }

    /**
     * Reports whether the current player has been defeated.
     *
     * <p>The engine's own verdict, not a count of what is left standing. It
     * fires a notification reading "&lt;player&gt; was defeated" on the same
     * transition, which is what pins the flag (wiki: policy-grading).
     *
     * @param engine The live engine instance.
     * @return True when the current player is out of the match.
     */
    static boolean isDefeated(Object engine) {
        return playerFlag(engine, EngineNames.PLAYER_DEFEATED);
    }

    /**
     * Reports whether the current player has been wiped out.
     *
     * <p>Stronger than defeat: nothing owned is left at all, and no ally holds
     * anything either. Its notification reads "&lt;player&gt; has been wiped
     * out".
     *
     * @param engine The live engine instance.
     * @return True when the current player holds nothing.
     */
    static boolean isWipedOut(Object engine) {
        return playerFlag(engine, EngineNames.PLAYER_WIPED);
    }

    /** Reads one boolean flag off the current player. */
    private static boolean playerFlag(Object engine, String field) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        if (team == null) {
            return false;
        }
        Object value = EngineAccess.readField(team, field);
        if (!(value instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: player field " + field + " is not a boolean" + EngineNames.PIN);
        }
        return ((Boolean) value).booleanValue();
    }

    /**
     * One player's scoreboard, as the engine keeps it.
     *
     * <p>Carried per player rather than for the local one alone, because the
     * question worth asking is comparative. "Our army is worth 3,400" says
     * nothing; "ours is 3,400 against a leader on 22,000" is the whole of the
     * match report.
     */
    static final class PlayerStat {

        private final int team;
        private final boolean local;
        private final boolean hostile;
        private final boolean defeated;
        private final boolean wiped;
        private final int income;
        private final int armyValue;
        private final int buildingValue;

        PlayerStat(
                int team,
                boolean local,
                boolean hostile,
                boolean defeated,
                boolean wiped,
                int income,
                int armyValue,
                int buildingValue) {
            this.team = team;
            this.local = local;
            this.hostile = hostile;
            this.defeated = defeated;
            this.wiped = wiped;
            this.income = income;
            this.armyValue = armyValue;
            this.buildingValue = buildingValue;
        }

        int team() {
            return this.team;
        }

        boolean local() {
            return this.local;
        }

        boolean hostile() {
            return this.hostile;
        }

        boolean defeated() {
            return this.defeated;
        }

        boolean wiped() {
            return this.wiped;
        }

        int income() {
            return this.income;
        }

        int armyValue() {
            return this.armyValue;
        }

        int buildingValue() {
            return this.buildingValue;
        }
    }

    /**
     * Returns the scoreboard for every player still holding a slot.
     *
     * <p>Read from the engine's own statistics rather than counted here. It
     * keeps income, army value and building value per player, charts all three,
     * and writes them into its own save file — so these are the figures the game
     * itself would show, and a reimplementation could disagree with them
     * (wiki: perception-visibility).
     *
     * <p>Absent slots are skipped and defeated ones are not: a player who has
     * just been eliminated is exactly who a run report wants to name, and their
     * final army value is the measurement that says whether we killed them or
     * somebody else did.
     *
     * @param engine The live engine instance.
     * @return One entry per occupied slot, in slot order.
     * @throws IllegalStateException When the roster or a statistic cannot be
     *     read, which is a pinned name that has moved.
     */
    static java.util.List<PlayerStat> playerStats(Object engine) {
        Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Object localTeam = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        Object roster;
        int size;
        try {
            roster = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER).get(null);
            size = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER_SIZE).getInt(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read the player roster" + EngineNames.PIN, e);
        }
        if (!(roster instanceof Object[])) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TEAM_ROSTER + " is not an array" + EngineNames.PIN);
        }
        Object[] slots = (Object[]) roster;
        java.util.List<PlayerStat> stats = new java.util.ArrayList<PlayerStat>();
        for (int index = 0; index < size && index < slots.length; index++) {
            Object player = slots[index];
            if (player == null || isAbsent(player)) {
                continue;
            }
            stats.add(
                    new PlayerStat(
                            EngineAccess.readIntField(player, EngineNames.TEAM_ID),
                            player == localTeam,
                            localTeam != null && Perception.isHostileBetween(localTeam, player),
                            EngineAccess.readBooleanField(player, EngineNames.PLAYER_DEFEATED),
                            EngineAccess.readBooleanField(player, EngineNames.PLAYER_WIPED),
                            statOf(player, EngineNames.STAT_INCOME),
                            statOf(player, EngineNames.STAT_ARMY_VALUE),
                            statOf(player, EngineNames.STAT_BUILDING_VALUE)));
        }
        return stats;
    }

    /**
     * Counts the occupied slots in the player roster.
     *
     * <p>The lobby's join detector: the roster is a static array on the team
     * class, filled as players connect, so a hosted lobby's "someone joined"
     * is a second non-absent slot — no engine reference and no live match
     * needed. Any failure to read counts as zero rather than throwing,
     * because the poller runs from boot and the roster may simply not exist
     * yet.
     */
    static int rosterCount() {
        Object roster;
        int size;
        try {
            Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
            roster = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER).get(null);
            size = EngineAccess.pinnedField(teamClass, EngineNames.TEAM_ROSTER_SIZE).getInt(null);
        } catch (IllegalAccessException | RuntimeException e) {
            return 0;
        }
        if (!(roster instanceof Object[])) {
            return 0;
        }
        Object[] slots = (Object[]) roster;
        int count = 0;
        for (int index = 0; index < size && index < slots.length; index++) {
            Object player = slots[index];
            if (player != null && !isAbsent(player)) {
                count++;
            }
        }
        return count;
    }

    /** Reports whether a player slot is empty rather than occupied. */
    private static boolean isAbsent(Object player) {
        Object answer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(player.getClass(), EngineNames.TEAM_ABSENT),
                        player);
        if (!(answer instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TEAM_ABSENT + "() did not return a boolean"
                            + EngineNames.PIN);
        }
        return ((Boolean) answer).booleanValue();
    }

    /**
     * Reads one named statistic for one player.
     *
     * <p>The constant is found by its own {@code name()} rather than by ordinal,
     * so a reordered enum fails to find the name instead of silently returning
     * the neighbouring statistic. {@code name()} is final on {@link Enum} and
     * returns a stored string, so nothing engine-side runs to answer it.
     *
     * @param player The player to measure.
     * @param constant The statistic's own constant name.
     * @return The figure.
     * @throws IllegalStateException When the enum carries no such constant, or
     *     the read does not return an int.
     */
    private static int statOf(Object player, String constant) {
        Class<?> statClass = EngineAccess.pinnedClass(EngineNames.PLAYER_STAT_CLASS);
        Object[] constants = statClass.getEnumConstants();
        if (constants == null) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.PLAYER_STAT_CLASS + " is no longer an enum"
                            + EngineNames.PIN);
        }
        for (Object candidate : constants) {
            if (!constant.equals(((Enum<?>) candidate).name())) {
                continue;
            }
            Object value =
                    EngineAccess.invoke(
                            EngineAccess.pinnedMethod(
                                    statClass,
                                    EngineNames.PLAYER_STAT_READ,
                                    EngineAccess.pinnedClass(EngineNames.TEAM_CLASS)),
                            candidate,
                            player);
            if (!(value instanceof Integer)) {
                throw new IllegalStateException(
                        "rw-agent: statistic " + constant + " did not return an int"
                                + EngineNames.PIN);
            }
            return ((Integer) value).intValue();
        }
        throw new IllegalStateException(
                "rw-agent: " + EngineNames.PLAYER_STAT_CLASS + " carries no constant named "
                        + constant + EngineNames.PIN);
    }

    /**
     * Returns how many players are still in the match.
     *
     * <p>Asked of the engine rather than counted here. It excludes absent,
     * defeated and wiped-out players, prints the same figure as "N players
     * remaining", and calls its own end-of-match hook when it reaches one -- so
     * this is the engine's scoreboard, and a reimplementation could disagree
     * with the thing that actually ends the game.
     *
     * @return The count of players still playing.
     */
    static int playersRemaining() {
        Class<?> teamClass = EngineAccess.pinnedClass(EngineNames.TEAM_CLASS);
        Object value =
                EngineAccess.invokeStatic(teamClass, EngineNames.PLAYERS_REMAINING);
        if (!(value instanceof Integer)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.PLAYERS_REMAINING
                            + "() did not return an int" + EngineNames.PIN);
        }
        return ((Integer) value).intValue();
    }
}
