package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Reflective snapshot of a live engine object.
 *
 * <p>{@code game-lib.jar} is obfuscated: the engine's base class carries some
 * 360 members of which exactly one kept a readable name, so identifying the
 * unit list by reading bytecode means correlating hundreds of single-letter
 * fields by type. The running object graph is the cheaper source of truth --
 * a list whose size matches the "there are 10 units on this map and 206 trees"
 * line the engine already printed identifies itself.
 *
 * <p>Reads only. Values are summarised structurally -- sizes, element classes,
 * runtime types -- and {@code toString} is called on nothing but strings,
 * primitives and their wrappers. Invoking arbitrary engine {@code toString}
 * implementations from a probe thread would be a side effect on a live
 * simulation, which is exactly what this must not be.
 */
final class Discovery {

    private static final int MAX_STRING = 120;

    private Discovery() {
    }

    /**
     * Describes every field of {@code target}, including inherited ones.
     *
     * @param target The object to describe. May be null.
     * @param label Caller-supplied tag identifying this snapshot.
     * @return A multi-line report, one field per line, most-derived class first.
     */
    static String describe(Object target, String label) {
        StringBuilder out = new StringBuilder();
        out.append("=== discovery ").append(label).append(" ===\n");
        if (target == null) {
            out.append("target is null\n");
            return out.toString();
        }
        out.append("target class: ").append(target.getClass().getName()).append('\n');

        for (Class<?> type = target.getClass(); type != null; type = type.getSuperclass()) {
            if (type == Object.class) {
                break;
            }
            out.append("--- ").append(type.getName()).append(" ---\n");
            for (Field field : declaredFieldsOf(type)) {
                out.append(describeField(target, field)).append('\n');
            }
        }
        return out.toString();
    }

    /** Returns a class's declared fields, or none when reflection is refused. */
    private static Field[] declaredFieldsOf(Class<?> type) {
        try {
            return type.getDeclaredFields();
        } catch (SecurityException e) {
            Log.error("cannot list fields of " + type.getName() + ": " + e);
            return new Field[0];
        }
    }

    /** Renders one field as {@code <modifiers> <name> : <type> = <summary>}. */
    private static String describeField(Object target, Field field) {
        String prefix =
                (Modifier.isStatic(field.getModifiers()) ? "static " : "       ")
                        + pad(field.getName())
                        + " : "
                        + pad(field.getType().getSimpleName())
                        + " = ";
        Object value;
        try {
            field.setAccessible(true);
            value = field.get(Modifier.isStatic(field.getModifiers()) ? null : target);
        } catch (ReflectiveOperationException | RuntimeException e) {
            // Reported rather than propagated: one unreadable field must not
            // abort a snapshot whose whole purpose is breadth.
            return prefix + "<unreadable: " + e.getClass().getSimpleName() + ">";
        }
        return prefix + summarise(value);
    }

    /** Summarises a value structurally, without invoking engine code. */
    private static String summarise(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof String) {
            String text = (String) value;
            String clipped = text.length() > MAX_STRING ? text.substring(0, MAX_STRING) + "..." : text;
            return "\"" + clipped + "\"";
        }
        if (value instanceof Number || value instanceof Boolean || value instanceof Character) {
            return value.toString();
        }
        if (value instanceof java.util.Collection) {
            java.util.Collection<?> collection = (java.util.Collection<?>) value;
            return value.getClass().getSimpleName() + " size=" + collection.size() + firstElement(collection);
        }
        if (value instanceof java.util.Map) {
            return value.getClass().getSimpleName() + " size=" + ((java.util.Map<?, ?>) value).size();
        }
        if (value.getClass().isArray()) {
            return value.getClass().getComponentType().getSimpleName()
                    + "[] len="
                    + java.lang.reflect.Array.getLength(value);
        }
        return value.getClass().getName();
    }

    /** Names the runtime class of a collection's first element, when it has one. */
    private static String firstElement(java.util.Collection<?> collection) {
        for (Object element : collection) {
            if (element != null) {
                return " of=" + element.getClass().getName();
            }
            return " of=null";
        }
        return "";
    }

    /** Right-pads for column alignment in the emitted report. */
    private static String pad(String text) {
        StringBuilder padded = new StringBuilder(text);
        while (padded.length() < 24) {
            padded.append(' ');
        }
        return padded.toString();
    }
}
