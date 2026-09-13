package rwbot.agent;

/**
 * Every obfuscated name for the engine's kill ledger, and nothing else.
 *
 * <p>Pure data, split out of {@link EngineNames} at its 600-line ceiling
 * along the seam {@link TypeNames} and {@link WaypointNames} left: these
 * names have one reader, {@link Scoreboard}, which reads each player's
 * kill tally beside the income and value statistics. The ledger is the
 * object behind the end-of-match "Units Killed" / "Units Lost" screen
 * ({@code gameFramework.f.e} builds it from exactly this object). Every kill
 * routes through its {@code a(killer, victim, damage)} from the damage path
 * ({@code game.f:560}), booking the killer's kill and the victim's loss in
 * one call, so both sides of a death agree. The per-loss trace only ever saw
 * OUR deaths; this is where the opponent's are.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> Every name
 * below moves between releases, silently — the jar is obfuscated, so a rename
 * produces a binding that resolves to nothing rather than a compile error.
 * {@link BindingCheck#verifyBindings()} resolves all of them against the jar
 * with no game running, and {@code make check} runs it, so a game update fails
 * at the gate rather than in a live run.
 */
final class LedgerNames {

    private LedgerNames() {
    }

    /** The kill ledger on the engine ({@code l.bY}), a {@link #LEDGER_CLASS}. */
    static final String LEDGER = "bY";

    /** The ledger's class, {@code gameFramework.bg}. */
    static final String LEDGER_CLASS = "com.corrodinggames.rts.gameFramework.bg";

    /**
     * {@link #LEDGER_CLASS}: one team's tally. Takes a
     * {@link EngineNames#TEAM_CLASS}, indexes by the team's own id, and
     * answers a fresh empty tally for a slot off the table.
     */
    static final String TALLY = "a";

    /** The tally's class, {@code gameFramework.bo}: public int counters. */
    static final String TALLY_CLASS = "com.corrodinggames.rts.gameFramework.bo";

    /** {@link #TALLY_CLASS}: mobile units this team has killed. */
    static final String UNITS_KILLED = "c";

    /** {@link #TALLY_CLASS}: buildings this team has killed. */
    static final String BUILDINGS_KILLED = "d";

    /** {@link #TALLY_CLASS}: mobile units this team has lost. */
    static final String UNITS_LOST = "f";

    /** {@link #TALLY_CLASS}: buildings this team has lost. */
    static final String BUILDINGS_LOST = "g";
}
