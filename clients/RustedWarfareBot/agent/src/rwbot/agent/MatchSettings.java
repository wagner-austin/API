package rwbot.agent;

/**
 * Writes on the engine's live settings object, for a seeded match only.
 *
 * <p>Split from {@link MatchSetup}, which queues the match and pins its
 * clocks; what lives here is the narrower job of assigning
 * {@code SettingsEngine} fields on the running engine. Both writes are
 * scoped to the seeded path on the same principle: a HOSTED match is a real
 * player's session, and neither their difficulty nor their autosave is this
 * agent's to take ({@link HostSetup}).
 */
final class MatchSettings {

    private MatchSettings() {
    }

    /**
     * Turns the engine's autosave off for this seeded match. Runs on the game
     * thread.
     *
     * <p>An autosave is pure local I/O no simulation state reads, so removing
     * it changes nothing a trace measures -- but writing the whole world to a
     * contended network filesystem freezes the engine long enough to breach
     * the planner's sample-read timeout, and nothing ever resumes a harness
     * match ({@link EngineNames#AUTOSAVING_FIELD}).
     */
    static void disableAutosaving() {
        Object engine = EngineHandle.current();
        Object settings = EngineAccess.readField(engine, EngineNames.SETTINGS_FIELD);
        EngineAccess.writeBooleanField(settings, EngineNames.AUTOSAVING_FIELD, false);
        Log.info("autosaving disabled; a harness match is never resumed");
    }

    /** Sets the AI difficulty on the live match. Runs on the game thread. */
    static void applyDifficulty(int difficulty) {
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
}
