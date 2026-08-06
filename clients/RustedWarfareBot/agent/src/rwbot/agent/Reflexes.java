package rwbot.agent;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Combat micro executed between samples, at the engine's own pace.
 *
 * <p>Seventy Impossible matches measured every shape the planner can express
 * and the human meta that beats the rung is none of them: it is hands --
 * artillery kited perpetually outside turret reach, units pulled back hurt
 * and healed, reactions faster than any sample interval (wiki:
 * community-play-strategies). The planner thinks once per lockstep interval
 * by design; this class is the hands underneath it, running on the game
 * thread every few ticks and issuing nothing but movement.
 *
 * <p>Two reflexes, both posture-gated by the planner over the wire:
 *
 * <ul>
 *   <li><b>Flee hurt</b> -- below the posture's health floor, with a threat
 *       in reach, walk directly away from it. Value preserved beats value
 *       spent: the engine runs the newest waypoint, so the planner's next
 *       order resumes the moment the reflex goes quiet.</li>
 *   <li><b>Kite</b> -- with a reach advantage over the nearest threat, hold
 *       the band between its reach and ours: step away whenever it closes
 *       inside its own reach plus a margin. The engine fires automatically
 *       in range, so kiting is pure movement.</li>
 * </ul>
 *
 * <p>The posture table is the planner's, sent once per type at match start:
 * the reach figures come from the catalogue the planner already owns, so the
 * agent never guesses a stat (wiki: mechanics-unit-catalogue). A type with
 * no entry has no reflexes and no reach, and a hostile of unknown reach is
 * treated as unarmed -- the margin absorbs the error.
 */
final class Reflexes {

    /** One type's standing orders. */
    static final class Posture {

        final float reach;
        final float speed;
        final boolean kite;
        final int hpFloor;

        Posture(float reach, float speed, boolean kite, int hpFloor) {
            this.reach = reach;
            this.speed = speed;
            this.kite = kite;
            this.hpFloor = hpFloor;
        }
    }

    /** Engine ticks between reflex passes: ~60ms of game time at 300fps. */
    static final int STRIDE_TICKS = 5;

    /** How far outside a threat's reach the kite band starts. */
    static final float KITE_MARGIN = 40f;

    /** The smallest reflex step worth issuing. */
    static final float MIN_STEP = 30f;

    /** The largest reflex step: a flee, or a badly overrun kite. */
    static final float MAX_STEP = 140f;

    private static final Map<String, Posture> POSTURES = new ConcurrentHashMap<String, Posture>();

    private Reflexes() {
    }

    /**
     * Stores one type's posture, replacing any previous one.
     *
     * @param type The unit type name, in the wire's vocabulary.
     * @param reach The type's attack range, from the planner's catalogue.
     * @param kite Whether owned units of the type hold the reach band.
     * @param hpFloor Percent of health below which owned units flee.
     */
    static void set(String type, float reach, float speed, boolean kite, int hpFloor) {
        POSTURES.put(type, new Posture(reach, speed, kite, hpFloor));
    }

    /** Forgets every posture. Selftest hygiene. */
    static void clear() {
        POSTURES.clear();
    }

    /** Reports whether any posture asks for anything. */
    static boolean armed() {
        for (Posture posture : POSTURES.values()) {
            if (posture.kite || posture.hpFloor > 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Runs one reflex pass over every owned unit with a posture.
     *
     * <p>Must run on the game thread. Movement only: a reflex never attacks,
     * never spends, never touches a structure -- the planner owns all of
     * that, and the engine running the newest waypoint is what hands a unit
     * back the moment the reflex goes quiet.
     *
     * @param engine The live engine instance.
     */
    static void step(Object engine) {
        if (POSTURES.isEmpty()) {
            return;
        }
        List<Object> hostiles = new java.util.ArrayList<Object>();
        for (Object entity : Perception.visibleEntities(engine)) {
            if (Perception.isHostileToLocalPlayer(engine, entity)) {
                hostiles.add(entity);
            }
        }
        if (hostiles.isEmpty()) {
            return;
        }
        for (Object unit : Perception.ownedUnits(engine)) {
            Posture posture = POSTURES.get(Perception.typeNameOf(unit));
            if (posture == null || !(posture.kite || posture.hpFloor > 0)) {
                continue;
            }
            reflex(engine, unit, posture, hostiles);
        }
    }

    /** Applies one unit's posture against the nearest threat. */
    private static void reflex(Object engine, Object unit, Posture posture, List<Object> hostiles) {
        float[] at = Perception.positionOf(unit);
        Object nearest = null;
        float best = Float.MAX_VALUE;
        for (Object hostile : hostiles) {
            float[] there = Perception.positionOf(hostile);
            float dx = there[0] - at[0];
            float dy = there[1] - at[1];
            float squared = dx * dx + dy * dy;
            if (squared < best) {
                best = squared;
                nearest = hostile;
            }
        }
        if (nearest == null) {
            return;
        }
        float distance = (float) Math.sqrt(best);
        float threatReach = reachOf(nearest);
        float[] there = Perception.positionOf(nearest);

        if (posture.hpFloor > 0 && distance <= threatReach + KITE_MARGIN) {
            float[] health = Perception.healthOf(unit);
            if (health[1] > 0f && health[0] / health[1] * 100f < posture.hpFloor) {
                stepAway(engine, unit, at, there, MAX_STEP);
                return;
            }
        }
        if (posture.kite
                && posture.reach > threatReach + KITE_MARGIN
                && posture.speed > speedOf(nearest)
                && distance < threatReach + KITE_MARGIN) {
            // Exactly out of the band plus a buffer, never a march -- and
            // only with a speed advantage, because both cuts without one
            // lost their A/Bs: a fixed 140-step put artillery into ~230
            // frames of silence per trigger, and computed steps against
            // faster chasers were a permanent shuffle that never fired.
            // Kite is a hover unit's move (log 2026-07-31).
            float needed = threatReach + KITE_MARGIN - distance + MIN_STEP;
            stepAway(engine, unit, at, there, Math.min(MAX_STEP, Math.max(MIN_STEP, needed)));
        }
    }

    /** Looks up a hostile's reach, treating an unknown type as unarmed. */
    private static float reachOf(Object hostile) {
        Posture posture = POSTURES.get(Perception.typeNameOf(hostile));
        return posture == null ? 0f : posture.reach;
    }

    /**
     * Looks up a hostile's speed, treating an unknown type as unchaseable.
     *
     * <p>Infinite rather than zero, deliberately: kite is gated on OUR
     * speed exceeding the chaser's, because a slower unit cannot out-step
     * what outruns it -- a 0.6-speed artillery piece stutter-stepping away
     * from 1.1-speed tanks traded a quarter of its worth for distance it
     * never gained (log 2026-07-31). An unknown chaser must therefore
     * read as faster, not slower, or the gate fails open.
     */
    private static float speedOf(Object hostile) {
        Posture posture = POSTURES.get(Perception.typeNameOf(hostile));
        return posture == null ? Float.MAX_VALUE : posture.speed;
    }

    /** Moves the given distance directly away from the threat. */
    private static void stepAway(
            Object engine, Object unit, float[] at, float[] there, float step) {
        float dx = at[0] - there[0];
        float dy = at[1] - there[1];
        float length = (float) Math.sqrt(dx * dx + dy * dy);
        if (length == 0f) {
            Orders.moveTo(engine, unit, at[0] + step, at[1]);
            return;
        }
        Orders.moveTo(engine, unit, at[0] + dx / length * step, at[1] + dy / length * step);
    }
}
