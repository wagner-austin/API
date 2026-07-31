package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * How the agent reaches an obfuscated engine, safely.
 *
 * <p>The mechanism, separate from the {@link EngineNames} table it resolves
 * and the {@link BindingCheck} guard that verifies it. This half barely
 * changes: reading a field reflectively is the same operation whether the field
 * is health or credits.
 *
 * <p>Every failure names the pinned build, because the overwhelmingly likely
 * cause of a missing field is a game update rather than a code bug, and a
 * reflection error that does not say so sends the reader hunting in the wrong
 * place.
 *
 * <p>Nothing here interprets what it reads. Reads with meaning attached are
 * {@link Perception}; writes are {@link Orders}.
 */
final class EngineAccess {

    private EngineAccess() {
    }

    static Iterable<?> entities() {
        Class<?> entityClass = pinnedClass(EngineNames.ENTITY_CLASS);
        Field field = pinnedField(entityClass, EngineNames.ENTITY_LIST);
        Object value;
        try {
            value = field.get(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("rw-agent: cannot read " + EngineNames.ENTITY_LIST + EngineNames.PIN, e);
        }
        if (!(value instanceof Iterable)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.ENTITY_CLASS + "." + EngineNames.ENTITY_LIST + " is not iterable" + EngineNames.PIN);
        }
        return (Iterable<?>) value;
    }

    static Class<?> pinnedClass(String binaryName) {
        try {
            return Class.forName(binaryName, false, Orders.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("rw-agent: class " + binaryName + " not found" + EngineNames.PIN, e);
        }
    }

    /**
     * Finds a pinned field, or reports that this class does not have one.
     *
     * <p>Distinct from {@link #pinnedField} because absence means different
     * things. A missing field on a class that should have it is drift and must
     * fail loudly; a field only some unit classes carry -- a production queue,
     * for instance -- is absent by design, and asking is the only way to tell.
     *
     * @param owner Class to search, including its superclasses.
     * @param name Obfuscated field name.
     * @return The accessible field, or null when this class has none.
     */
    static Field fieldIfPresent(Class<?> owner, String name) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                Field field = type.getDeclaredField(name);
                field.setAccessible(true);
                return field;
            } catch (NoSuchFieldException e) {
                continue;
            }
        }
        return null;
    }

    static Field pinnedField(Class<?> owner, String name) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                Field field = type.getDeclaredField(name);
                field.setAccessible(true);
                return field;
            } catch (NoSuchFieldException e) {
                continue;
            }
        }
        throw new IllegalStateException(
                "rw-agent: field " + name + " not found on " + owner.getName() + EngineNames.PIN);
    }

    static Method pinnedMethod(Class<?> owner, String name, Class<?>... parameters) {
        for (Class<?> type = owner; type != null; type = type.getSuperclass()) {
            try {
                Method method = type.getDeclaredMethod(name, parameters);
                method.setAccessible(true);
                return method;
            } catch (NoSuchMethodException e) {
                continue;
            }
        }
        throw new IllegalStateException(
                "rw-agent: method "
                        + name
                        + java.util.Arrays.toString(parameters)
                        + " not found on "
                        + owner.getName()
                        + EngineNames.PIN);
    }

    static Object readField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).get(target);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("rw-agent: cannot read " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Reads an {@code int} field through the same pinned-name machinery as
     * every other engine read, so a moved name fails identically here.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not an int.
     */
    static int readIntField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getInt(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read int " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Writes an {@code int} field through the same pinned-name machinery.
     *
     * <p>Exists for exactly one caller so far: the match setup zeroing the
     * frame and clock counters on the match's first live tick, which is the
     * engine's own new-game convention performed at the moment the engine's
     * own path skips it (wiki: policy-determinism).
     *
     * @param target Object to write to.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @param value The value to store.
     * @throws IllegalStateException When the field is absent or not an int.
     */
    static void writeIntField(Object target, String name, int value) {
        try {
            pinnedField(target.getClass(), name).setInt(target, value);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot write int " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Writes a {@code float} field through the same pinned-name machinery.
     *
     * @param target Object to write to.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @param value The value to store.
     * @throws IllegalStateException When the field is absent or not a float.
     */
    static void writeFloatField(Object target, String name, float value) {
        try {
            pinnedField(target.getClass(), name).setFloat(target, value);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot write float " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Writes a {@code boolean} field through the same pinned-name machinery.
     *
     * @param target Object to write to.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @param value The value to store.
     * @throws IllegalStateException When the field is absent or not a boolean.
     */
    static void writeBooleanField(Object target, String name, boolean value) {
        try {
            pinnedField(target.getClass(), name).setBoolean(target, value);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot write boolean " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Reads a {@code long} field through the same pinned-name machinery.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not a long.
     */
    static long readLongField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getLong(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read long " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Reads a {@code double} field through the same pinned-name machinery.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not a double.
     */
    static double readDoubleField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getDouble(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read double " + name + EngineNames.PIN, e);
        }
    }

    /**
     * Reads a {@code boolean} field through the same pinned-name machinery.
     *
     * @param target Object to read from.
     * @param name Obfuscated field name, pinned to the recorded build.
     * @return The field value.
     * @throws IllegalStateException When the field is absent or not a boolean.
     */
    static boolean readBooleanField(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getBoolean(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read boolean " + name + EngineNames.PIN, e);
        }
    }

    static float readFloat(Object target, String name) {
        try {
            return pinnedField(target.getClass(), name).getFloat(target);
        } catch (IllegalAccessException | IllegalArgumentException e) {
            throw new IllegalStateException("rw-agent: cannot read float " + name + EngineNames.PIN, e);
        }
    }

    static Object invoke(Method method, Object target, Object... arguments) {
        try {
            return method.invoke(target, arguments);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("rw-agent: call to " + method.getName() + " failed", e);
        }
    }

    static Object invokeStatic(Class<?> owner, String name) {
        return invoke(pinnedMethod(owner, name), null);
    }
}
