package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * How to look at a live engine object safely, and how to render what is found.
 *
 * <p>The shared vocabulary behind {@link Snapshot} and {@link GraphSearch}.
 * Neither of those decides how a value is summarised or which fields belong to
 * an object; both ask here, which is what keeps two traversals from drifting
 * into two different answers.
 *
 * <p><b>Two rules hold everything here together.</b> Values are summarised
 * structurally and {@code toString} is called on nothing but strings,
 * primitives and their wrappers, because invoking arbitrary engine
 * {@code toString} from a probe thread would be a side effect on a live
 * simulation. And every hierarchy walk stops at the platform boundary: the
 * boundary was applied in one walk and missed in another twice, so
 * {@link #ownedHierarchy} exists to make it structural rather than something
 * each new traversal has to remember.
 */
final class ObjectView {

    private ObjectView() {
    }

    static final int MAX_STRING = 120;

    /**
     * Returns the values a container holds, or null when the object is not one.
     *
     * <p>Arrays are included: {@code java.lang.reflect.Array} reads them through
     * a supported API rather than through their fields, which they do not have.
     *
     * @param value The object to inspect.
     * @return Its contained values, or null when it is not a container.
     */
    static java.util.Collection<?> containedValues(Object value) {
        if (value instanceof java.util.Collection) {
            return (java.util.Collection<?>) value;
        }
        if (value instanceof java.util.Map) {
            return ((java.util.Map<?, ?>) value).values();
        }
        if (value.getClass().isArray() && !value.getClass().getComponentType().isPrimitive()) {
            int length = java.lang.reflect.Array.getLength(value);
            java.util.List<Object> elements = new java.util.ArrayList<Object>(length);
            for (int i = 0; i < length; i++) {
                elements.add(java.lang.reflect.Array.get(value, i));
            }
            return elements;
        }
        return null;
    }

    /** Returns the class of a collection's or array's first non-null element. */
    static String elementClassOf(Object value) {
        if (value instanceof java.util.Collection) {
            for (Object element : (java.util.Collection<?>) value) {
                if (element != null) {
                    return element.getClass().getName();
                }
            }
            return null;
        }
        if (value.getClass().isArray()) {
            int length = java.lang.reflect.Array.getLength(value);
            for (int i = 0; i < length; i++) {
                Object element = java.lang.reflect.Array.get(value, i);
                if (element != null) {
                    return element.getClass().getName();
                }
            }
        }
        return null;
    }

    /** Reads a field, yielding null rather than propagating a reflection failure. */
    static Object readQuietly(Object owner, Field field) {
        try {
            field.setAccessible(true);
            return field.get(Modifier.isStatic(field.getModifiers()) ? null : owner);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /**
     * Reports whether a class belongs to the Java platform rather than the game.
     *
     * <p>Reflecting into platform internals is both useless here and actively
     * hostile to the module system: it warns today and is denied in later JDKs.
     * The engine's own classes all sit under {@code com.corrodinggames}.
     *
     * @param type Class to test.
     * @return True for platform classes and for array and primitive types.
     */
    static boolean isPlatformClass(Class<?> type) {
        if (type.isPrimitive() || type.isArray()) {
            return true;
        }
        Package declared = type.getPackage();
        if (declared == null) {
            return false;
        }
        String name = declared.getName();
        return name.startsWith("java.")
                || name.startsWith("javax.")
                || name.startsWith("jdk.")
                || name.startsWith("sun.")
                || name.startsWith("com.sun.");
    }

    /** Finds a declared field by name anywhere in the class hierarchy. */
    static Field findField(Class<?> type, String name) {
        for (Class<?> current = type; current != null; current = current.getSuperclass()) {
            try {
                return current.getDeclaredField(name);
            } catch (NoSuchFieldException e) {
                continue;
            }
        }
        return null;
    }

    /**
     * The classes whose declared fields belong to a game object: the class
     * itself and its superclasses, stopping at {@code Object} and at the first
     * platform class.
     *
     * <p>Every hierarchy walk goes through here, and that is the point. The
     * platform boundary had been applied in one walk and missed in another
     * twice: first when containers were traversed by their declared fields,
     * then again here, where a game class extending a JDK type -- twelve extend
     * {@code Thread}, four extend {@code Exception}, and the master entity list
     * type extends {@code AbstractList} -- had its superclass internals
     * reflected. A shared helper makes the boundary structural instead of
     * something each new walk has to remember.
     *
     * @param type Most-derived class to start from.
     * @return The owned classes, most-derived first; empty when {@code type} is
     *     itself a platform class.
     */
    static java.util.List<Class<?>> ownedHierarchy(Class<?> type) {
        java.util.List<Class<?>> chain = new java.util.ArrayList<Class<?>>();
        for (Class<?> current = type;
                current != null && current != Object.class && !isPlatformClass(current);
                current = current.getSuperclass()) {
            chain.add(current);
        }
        return chain;
    }

    /** Returns a class's declared fields, or none when reflection is refused. */
    static Field[] declaredFieldsOf(Class<?> type) {
        try {
            return type.getDeclaredFields();
        } catch (SecurityException e) {
            Log.error("cannot list fields of " + type.getName() + ": " + e);
            return new Field[0];
        }
    }

    /** Renders one field as {@code <modifiers> <name> : <type> = <summary>}. */
    static String describeField(Object target, Field field) {
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
    static String summarise(Object value) {
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
        // Enums by constant name. This does not break the no-engine-toString
        // rule: Enum.name() is final on java.lang.Enum and returns the stored
        // constant name, so no engine code runs. Without it a state field
        // renders as its class and the value -- the whole point of reading it
        // -- is lost, which is exactly what the first AI zone dump hit.
        if (value instanceof Enum) {
            return value.getClass().getSimpleName() + "." + ((Enum<?>) value).name();
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
    static String firstElement(java.util.Collection<?> collection) {
        for (Object element : collection) {
            if (element != null) {
                return " of=" + element.getClass().getName();
            }
            return " of=null";
        }
        return "";
    }

    /** Right-pads for column alignment in the emitted report. */
    static String pad(String text) {
        StringBuilder padded = new StringBuilder(text);
        while (padded.length() < 24) {
            padded.append(' ');
        }
        return padded.toString();
    }
}
