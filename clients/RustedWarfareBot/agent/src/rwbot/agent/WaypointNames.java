package rwbot.agent;

/**
 * Every obfuscated name for a unit's waypoint queue, and nothing else.
 *
 * <p>Pure data, split out of {@link EngineNames} when that table crossed its
 * 600-line ceiling — the same seam {@link TypeNames} left along: these names
 * have exactly one reader, {@link BuildWatch}, which watches a dispatched
 * build order's waypoint to tell "in flight" from "the engine dropped it".
 * The world-sampling names stay in {@link EngineNames}; the two are read by
 * different callers and grow for different reasons.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> Every name
 * below moves between releases, silently — the jar is obfuscated, so a rename
 * produces a binding that resolves to nothing rather than a compile error.
 * {@link BindingCheck#verifyBindings()} resolves all of them against the jar
 * with no game running, and {@code make check} runs it, so a game update fails
 * at the gate rather than in a live run.
 */
final class WaypointNames {

    private WaypointNames() {
    }

    /**
     * A unit's waypoint queue: the array ({@code y.g}) and its live length
     * ({@code y.f}).
     *
     * <p>Read, never written: {@link BuildWatch} watches a dispatched build
     * order's waypoint to tell "in flight" from "the engine dropped it".
     * The engine's own accessors read the same pair ({@code y.ar()} returns
     * {@code g[0]} when {@code f > 0}).
     */
    static final String WAYPOINT_ARRAY = "g";

    /** See {@link #WAYPOINT_ARRAY}. */
    static final String WAYPOINT_COUNT = "f";

    /**
     * The waypoint class itself ({@code au}) — the element type of
     * {@code y.g}. Never instantiated or read by name at runtime (the
     * instances come out of the array), but pinned so the gate can resolve
     * the fields below against the jar with no game running.
     */
    static final String WAYPOINT_CLASS = "com.corrodinggames.rts.game.units.au";

    /**
     * Fields of one waypoint ({@code au}): its kind, its build type, and its
     * target coordinates.
     *
     * <p>The engine's own resume-check compares exactly these --
     * {@code au2.a != av.c || au2.b != as2 || |au2.e - x| >= 10 || ...}
     * ({@code y.java:3408}) -- so matching a dispatched order to a queued
     * waypoint by (kind, type reference, coordinates) is the engine's own
     * identity rule, not an invented one.
     */
    static final String WAYPOINT_KIND = "a";

    /** See {@link #WAYPOINT_KIND}. */
    static final String WAYPOINT_BUILD_TYPE = "b";

    /** See {@link #WAYPOINT_KIND}. */
    static final String WAYPOINT_X = "e";

    /** See {@link #WAYPOINT_KIND}. */
    static final String WAYPOINT_Y = "f";

    /** The waypoint-kind enum ({@code av}) and its build member ({@code av.c}). */
    static final String WAYPOINT_KIND_CLASS = "com.corrodinggames.rts.game.units.av";

    /** See {@link #WAYPOINT_KIND_CLASS}. */
    static final String WAYPOINT_KIND_BUILD = "c";
}
