package rwbot.agent;

import java.lang.reflect.Method;

/**
 * Reflective handle on the live engine singleton.
 *
 * <p>The engine narrates its own construction -- "Created new gameEngine
 * of:com.corrodinggames.rts.game.i" -- and that instance is reachable through
 * a static accessor on its base class, {@code gameFramework.l.B()}, whose whole
 * body is {@code return al;}. Being a pure field read is what makes it safe to
 * call from a probe thread: it cannot advance or mutate the simulation.
 *
 * <p>Reached reflectively rather than by compiling against the game jar. The
 * agent then builds without the 451 MB pinned tree on the path, and a name that
 * moved in a game update fails here with a message naming the pinned build
 * instead of as a compile error in an unrelated place.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b>
 */
final class EngineHandle {

    private static final String ENGINE_CLASS = "com.corrodinggames.rts.gameFramework.l";
    private static final String ACCESSOR = "B";

    private EngineHandle() {
    }

    /**
     * Returns the live engine instance.
     *
     * @return The engine singleton, or null before it has been constructed.
     * @throws IllegalStateException When the class or accessor is absent, which
     *     means the obfuscated names moved and the pin is stale.
     */
    static Object current() {
        Class<?> engineClass;
        try {
            engineClass = Class.forName(ENGINE_CLASS, false, EngineHandle.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(
                    "rw-agent: engine class " + ENGINE_CLASS + " not found; the pinned build is"
                            + " 1.15 (code 176, build #28) and obfuscated names change between"
                            + " releases",
                    e);
        }

        Method accessor;
        try {
            accessor = engineClass.getDeclaredMethod(ACCESSOR);
        } catch (NoSuchMethodException e) {
            throw new IllegalStateException(
                    "rw-agent: engine accessor " + ENGINE_CLASS + "." + ACCESSOR + "() not found;"
                            + " re-derive it against the pinned jar",
                    e);
        }

        try {
            accessor.setAccessible(true);
            return accessor.invoke(null);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(
                    "rw-agent: engine accessor " + ENGINE_CLASS + "." + ACCESSOR + "() failed", e);
        }
    }
}
