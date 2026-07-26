package rwbot.agent;

/**
 * What the agent was asked to do, as an immutable value.
 *
 * <p>This type only holds and exposes; turning a {@code -javaagent:<jar>=...}
 * argument into one is {@link AgentOptionsParser}. Splitting them keeps the
 * question "what can the agent be asked to do" answerable without reading a
 * parser, and every accessor here documents why an option exists rather than
 * what it is spelled.
 *
 * <p>Array-valued options are returned as copies. A caller that mutated a
 * returned array would silently change what a later reader sees, and these are
 * read from more than one thread.
 */
final class AgentOptions {

    private final int[] discoverAtSeconds;
    private final boolean exitAfterDiscovery;
    private final String[] inspectFields;
    private final String findElementsUnder;
    private final String stateOutPath;
    private final String typeFlagsPath;
    private final int orderMoveAtSeconds;
    private final float[] orderMoveBy;
    private final int orderMoveUnitIndex;
    private final String buildType;
    private final int channelPort;
    private final int sampleIntervalMs;

    AgentOptions(
            int[] discoverAtSeconds,
            boolean exitAfterDiscovery,
            String[] inspectFields,
            String findElementsUnder,
            int orderMoveAtSeconds,
            float[] orderMoveBy,
            int orderMoveUnitIndex,
            String buildType,
            String stateOutPath,
            String typeFlagsPath,
            int channelPort,
            int sampleIntervalMs) {
        this.discoverAtSeconds = discoverAtSeconds;
        this.exitAfterDiscovery = exitAfterDiscovery;
        this.inspectFields = inspectFields;
        this.findElementsUnder = findElementsUnder;
        this.orderMoveAtSeconds = orderMoveAtSeconds;
        this.orderMoveBy = orderMoveBy;
        this.orderMoveUnitIndex = orderMoveUnitIndex;
        this.buildType = buildType;
        this.stateOutPath = stateOutPath;
        this.typeFlagsPath = typeFlagsPath;
        this.channelPort = channelPort;
        this.sampleIntervalMs = sampleIntervalMs;
    }

    /**
     * Loopback port the planner connects to, or 0 when the channel is off.
     *
     * <p>Off by default. A probe run and a planner-driven run are different
     * things, and opening a listening socket as a side effect of attaching the
     * agent would make every discovery run also a server.
     */
    int channelPort() {
        return channelPort;
    }

    /** True when a planner channel was requested. */
    boolean channelRequested() {
        return channelPort > 0;
    }

    /**
     * Milliseconds between world samples pushed to the planner.
     *
     * <p>Defaults to 250 ms -- four decisions a second. An RTS does not need
     * per-tick decisions, and the sampling rate is what keeps a cross-process
     * planner viable (wiki: runtime-split-java-agent-python-brain).
     */
    int sampleIntervalMs() {
        return sampleIntervalMs;
    }

    /**
     * Unit-type name to construct instead of moving, or empty to move.
     *
     * <p>Shares the timer and roster index with the move probe because the two
     * differ only in the verb: same subject, same destination, different
     * command setter. Keeping them one option set makes that similarity
     * visible rather than duplicating the scheduling.
     */
    String buildType() {
        return buildType;
    }

    /**
     * Index into the owned-entity roster of the unit to order.
     *
     * <p>The agent does not decide which unit is worth moving, or even which
     * can move: it publishes the roster and dispatches against an index.
     * Choosing is the planner's job, and a mobility predicate guessed here
     * would be exactly the decision logic the agent must not hold (wiki:
     * multiplayer-portability-invariants).
     */
    int orderMoveUnitIndex() {
        return orderMoveUnitIndex;
    }

    /**
     * Elapsed time at which to order the player's first unit to move, or 0 when
     * not requested.
     *
     * <p>A time rather than a flag because the order is only meaningful once a
     * map has loaded and the player owns something; issuing at boot would
     * report "no unit" and prove nothing.
     */
    int orderMoveAtSeconds() {
        return orderMoveAtSeconds;
    }

    /** True when a move order was requested. */
    boolean orderRequested() {
        return orderMoveAtSeconds > 0;
    }

    /**
     * World-space offset the ordered unit is sent by, as {x, y}.
     *
     * <p>An offset rather than an absolute point: the destination has to be
     * somewhere the unit can actually reach, and the only position known to be
     * on reachable terrain is the one the unit already occupies.
     */
    float[] orderMoveBy() {
        return orderMoveBy.clone();
    }

    /**
     * Binary-name prefix to search the live object graph for, or empty when not
     * requested.
     */
    String findElementsUnder() {
        return findElementsUnder;
    }

    /**
     * Absolute path the NDJSON world stream is appended to, or empty when not
     * requested. Written at each discovery offset, so it shares that schedule
     * rather than inventing a second one.
     */
    String stateOutPath() {
        return stateOutPath;
    }

    /**
     * Absolute path the unit-type placement flags are written to, or empty when
     * not requested.
     *
     * <p>Written once, at the first discovery offset, because the answer cannot
     * change during a match: unit types are loaded with the assets and the mod
     * set is fixed at boot. Shares the discovery schedule for the same reason
     * the state dump does -- both need a loaded game and neither needs its own
     * timer.
     */
    String typeFlagsPath() {
        return typeFlagsPath;
    }

    /**
     * Engine field names whose elements to expand, in declaration order.
     *
     * <p>A whole-object snapshot reports that a collection holds eleven things;
     * it cannot say what distinguishes them. Naming a field opts into one extra
     * level of depth for that field only, which keeps the default snapshot
     * cheap and the expensive read deliberate.
     */
    String[] inspectFields() {
        return inspectFields.clone();
    }

    /**
     * Whether to halt the JVM once the last snapshot has been emitted.
     *
     * <p>Off by default, and named rather than implied: a probe that killed the
     * game as a side effect of asking for a snapshot would be a surprise the
     * first time someone combined discovery with a longer run.
     */
    boolean exitAfterDiscovery() {
        return exitAfterDiscovery;
    }

    /**
     * Elapsed times, in seconds, at which to dump a discovery snapshot.
     *
     * <p>Empty when discovery was not requested. Multiple times are the point:
     * comparing a snapshot taken before a map loads with one taken after
     * identifies which fields hold match state, which a single dump cannot.
     */
    int[] discoverAtSeconds() {
        return discoverAtSeconds.clone();
    }

    /** True when any discovery snapshot was requested. */
    boolean discoveryRequested() {
        return discoverAtSeconds.length > 0;
    }
}
