package rwbot.agent;

/**
 * Watches every dispatched build order until the engine's verdict is visible.
 *
 * <p>The engine refuses a blocked placement by doing NOTHING a planner can
 * see: the construction attempt creates a ghost of the building, runs the
 * blocked-placement pair on it, destroys the ghost, and removes the build
 * waypoint after one in-range attempt ({@code y.java:4852} and the
 * {@code if (!z2.c) this.ay()} arm of the build-waypoint handler). At the
 * shipped log level none of its diagnostics reach the log. The planner's only
 * recourse used to be a 45-sample quiet clock; the Hard-rung panel measured
 * what that costs (wiki log 2026-08-31, verdict-withheld).
 *
 * <p>So this class DETECTS the verdict instead of predicting it. Prediction
 * -- mirroring the ghost-and-blocked-test dance through reflection -- would
 * be a second copy of an obfuscated rule, wrong the day the engine's copy
 * moves. Detection reads the engine's own outcome: a dispatched build whose
 * waypoint has vanished without its structure appearing IS a refusal, and the
 * waypoint-matching rule used here (kind, type reference, coordinates) is the
 * engine's own resume-check identity ({@code y.java:3408}).
 *
 * <p>Every method runs on the game thread -- dispatch and the sample sweep
 * both do -- so the pending list needs no lock.
 */
final class BuildWatch {

    private BuildWatch() {
    }

    /** One dispatched build order awaiting the engine's verdict. */
    static final class Pending {

        final long builderId;
        final String typeName;
        final float x;
        final float y;
        final int frame;
        boolean seenWaypoint;

        Pending(long builderId, String typeName, float x, float y, int frame) {
            this.builderId = builderId;
            this.typeName = typeName;
            this.x = x;
            this.y = y;
            this.frame = frame;
        }
    }

    /**
     * Frames a dispatched order may stay unseen in the builder's queue before
     * never-appearing counts as refusal. The command crosses the controller
     * queue in a few frames; ninety (over a full 75-frame lockstep window)
     * separates queue latency from a waypoint the validator rejected.
     */
    static final int GRACE_FRAMES = 90;

    /**
     * How far a standing structure may sit from the ordered point and still
     * be that order's outcome. The engine snaps placement to its tile grid
     * ({@code l2.bL.b(...)} in the construction attempt), so the built
     * position differs from the asked one by up to a tile.
     */
    static final float SITE_TOLERANCE = 32.0f;

    /**
     * How close a queued waypoint must be to the ordered point to be that
     * order's waypoint. Ten world units, because that is the engine's own
     * resume-check tolerance ({@code y.java:3408}).
     */
    static final float WAYPOINT_TOLERANCE = 10.0f;

    private static final java.util.List<Pending> PENDING = new java.util.ArrayList<Pending>();

    /** Starts watching one just-dispatched build order. */
    static void record(long builderId, String typeName, float x, float y, int frame) {
        PENDING.add(new Pending(builderId, typeName, x, y, frame));
    }

    /** Forgets every watched order. Between matches, not between samples. */
    static void reset() {
        PENDING.clear();
    }

    /**
     * Sweeps the watch list against the live world.
     *
     * @param engine The live engine instance.
     * @param frame The engine's frame counter at the sweep.
     * @return The orders the engine has refused, oldest first. Each is
     *     removed from the watch; so are orders that RESOLVED (their
     *     structure stands) and orders whose builder is gone -- a dead
     *     builder's order died with it, which the roster already reports,
     *     and calling it a site refusal would poison a fine site.
     */
    static java.util.List<Pending> sweep(Object engine, int frame) {
        java.util.List<Pending> refused = new java.util.ArrayList<Pending>();
        java.util.Iterator<Pending> pending = PENDING.iterator();
        while (pending.hasNext()) {
            Pending order = pending.next();
            if (structureStandsAt(engine, order)) {
                pending.remove();
                continue;
            }
            Object builder = Perception.findOwnedById(engine, order.builderId);
            if (builder == null) {
                pending.remove();
                continue;
            }
            if (hasBuildWaypoint(builder, order)) {
                order.seenWaypoint = true;
                continue;
            }
            if (order.seenWaypoint || frame - order.frame > GRACE_FRAMES) {
                refused.add(order);
                pending.remove();
            }
        }
        return refused;
    }

    /**
     * Reports whether the order's structure is standing near its site.
     *
     * <p>Complete or not: a rising structure is already the order succeeding,
     * and the roster carries it either way.
     */
    private static boolean structureStandsAt(Object engine, Pending order) {
        for (Object entity : Perception.ownedUnits(engine)) {
            if (!order.typeName.equals(Perception.typeNameOf(entity))) {
                continue;
            }
            float[] position = Perception.positionOf(entity);
            if (Math.abs(position[0] - order.x) <= SITE_TOLERANCE
                    && Math.abs(position[1] - order.y) <= SITE_TOLERANCE) {
                return true;
            }
        }
        return false;
    }

    /**
     * Reports whether the builder still queues this order's build waypoint.
     *
     * <p>Kind, type reference and coordinates -- the engine's own identity
     * rule for matching an order to a waypoint. The type is resolved through
     * the same registry the dispatch used, so the reference comparison is the
     * comparison the engine itself makes.
     */
    private static boolean hasBuildWaypoint(Object builder, Pending order) {
        Object array = EngineAccess.readField(builder, WaypointNames.WAYPOINT_ARRAY);
        int count = EngineAccess.readIntField(builder, WaypointNames.WAYPOINT_COUNT);
        if (array == null) {
            return false;
        }
        Object buildKind =
                EngineAccess.readStaticField(
                        EngineAccess.pinnedClass(WaypointNames.WAYPOINT_KIND_CLASS),
                        WaypointNames.WAYPOINT_KIND_BUILD);
        Object type = Orders.resolveType(order.typeName);
        int length = java.lang.reflect.Array.getLength(array);
        for (int index = 0; index < count && index < length; index++) {
            Object waypoint = java.lang.reflect.Array.get(array, index);
            if (waypoint == null) {
                continue;
            }
            if (EngineAccess.readField(waypoint, WaypointNames.WAYPOINT_KIND) != buildKind) {
                continue;
            }
            if (EngineAccess.readField(waypoint, WaypointNames.WAYPOINT_BUILD_TYPE) != type) {
                continue;
            }
            float wx = EngineAccess.readFloat(waypoint, WaypointNames.WAYPOINT_X);
            float wy = EngineAccess.readFloat(waypoint, WaypointNames.WAYPOINT_Y);
            if (Math.abs(wx - order.x) < WAYPOINT_TOLERANCE
                    && Math.abs(wy - order.y) < WAYPOINT_TOLERANCE) {
                return true;
            }
        }
        return false;
    }
}
