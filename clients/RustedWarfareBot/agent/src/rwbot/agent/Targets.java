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
}
