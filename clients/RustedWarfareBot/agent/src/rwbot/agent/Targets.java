package rwbot.agent;

/**
 * The methods the agent neutralises to keep the engine alive without a display.
 *
 * <p>Every entry here is a GUI-rendering callback that dereferences a field
 * which is only populated when a real {@code org.newdawn.slick.Graphics} exists.
 * Headless there is none, so the callback throws on its first invocation and
 * takes the game thread down through {@code uncaughtExceptionHandler}.
 *
 * <p>These are render-path only. Nothing here touches simulation state, so
 * neutralising them cannot alter the deterministic tick -- which is what keeps
 * the work multiplayer-legal (see wiki: multiplayer-portability-invariants).
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> The jar is
 * obfuscated and these names change silently between releases. {@link Premain}
 * fails loudly if a listed class is never seen, rather than booting into a
 * silently unpatched engine.
 */
final class Targets {

    private Targets() {
    }

    /**
     * Internal class name to the methods no-opped inside it.
     *
     * <p>A method may be listed bare ({@code "EnableScissorRegion"}, matching
     * every overload) or with its descriptor appended
     * ({@code "EnableScissorRegion(Z)V"}) to pin one overload exactly.
     */
    static java.util.Map<String, java.util.Set<String>> byClass() {
        java.util.Map<String, java.util.Set<String>> targets =
                new java.util.LinkedHashMap<String, java.util.Set<String>>();

        // com.corrodinggames.rts.java.d.a -- the LibRocket render backend.
        //
        // EnableScissorRegion(Z) is a JNI callback: native com.LibRocket.render
        // calls back into it on the first in-game GUI frame. It does exactly
        // two things -- Graphics.setWorldClip / clearWorldClip on field `j`,
        // and it maintains the boolean flag `h`. Field `j` is null headless,
        // so it throws at once.
        //
        // A bare `return` is deliberate rather than lazy. Flag `h` has exactly
        // three references in the class: the two writes in this method, and one
        // read in RenderGeometryPossiblyCompiled that guards a branch
        // dereferencing field `g`. Leaving `h` permanently false is therefore
        // strictly safer than preserving `h = enabled` would be -- preserving it
        // would arm a second null dereference rather than avoid one.
        targets.put(
                "com/corrodinggames/rts/java/d/a",
                new java.util.LinkedHashSet<String>(
                        java.util.Arrays.asList("EnableScissorRegion(Z)V")));

        return targets;
    }

    /**
     * The unit class carrying the wall-paced cosmetic spawners.
     *
     * <p>Named separately from {@link #byClass()} because these are NOT
     * render-path no-ops: they run inside the simulation tick, and
     * neutralising them changes which draws the simulation consumes. That is
     * the point, and it is also why {@link Premain} skips them when hosting
     * -- the same containment {@link SyncPathTransformer} gets, for the same
     * reason (wiki: multiplayer-portability-invariants).
     */
    static final String EFFECT_SPAWNER_CLASS = "com/corrodinggames/rts/game/units/y";

    /**
     * Methods whose only body is a wall-clock-paced cosmetic effect spawn.
     *
     * <p><b>Each spends two {@code Math.random()} draws on a schedule the
     * frame delta sets.</b> Both have the identical shape, verified against
     * the pinned jar's bytecode:
     *
     * <pre>
     *   U = f.a(U, delta)              // per-unit accumulator, MEASURED delta
     *   if (U == 0) { U = 5.0f;
     *       if (s_()) { ...
     *           x = am.eo + (-8.0 + Math.random() * 16.0)   // y.a:11504
     *           y = am.ep + (-8.0 + Math.random() * 16.0)   // y.a:11505
     *           ... spawn effect, set its velocity/life/colour ... }}
     *   return
     * </pre>
     *
     * <p>Everything after the gate writes only to the effect object the
     * particle manager returned -- {@code P/Q} velocity, {@code V/W} life,
     * {@code E/F/G} colour, the {@code r} flag -- and to the unit's own
     * accumulator. No unit state, no world state. So the simulation loses
     * nothing and the draw stream stops depending on the wall clock.
     *
     * <p><b>Found by measurement, not by reading.</b> The draw tap named
     * {@code y.a:11504} and {@code :11505} as the first call site whose
     * per-window count differed between two invocations, at frame 2850, with
     * every downstream difference following from it. Same pattern as the
     * ambient spawner already silenced, on the unit class rather than the
     * effects manager (wiki log 2026-08-30).
     */
    static java.util.Set<String> effectSpawners() {
        return new java.util.LinkedHashSet<String>(
                java.util.Arrays.asList(
                        "a(Lcom/corrodinggames/rts/game/units/am;FI)V",
                        "b(Lcom/corrodinggames/rts/game/units/am;FI)V"));
    }
}
