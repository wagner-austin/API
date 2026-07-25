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

        for (Class<?> type : ownedHierarchy(target.getClass())) {
            out.append("--- ").append(type.getName()).append(" ---\n");
            for (Field field : declaredFieldsOf(type)) {
                out.append(describeField(target, field)).append('\n');
            }
        }
        return out.toString();
    }

    /**
     * Expands one named field's elements, one level deep.
     *
     * <p>A whole-object snapshot can say a collection holds eleven things but
     * not what distinguishes them. This answers that for a single field, which
     * is what turns "probably the unit list" into a decided question.
     *
     * @param target The object owning the field.
     * @param fieldName Declared field name, searched up the class hierarchy.
     * @return A multi-line report of each element's class and own fields.
     */
    static String describeElements(Object target, String fieldName) {
        StringBuilder out = new StringBuilder();
        out.append("=== elements of ").append(fieldName).append(" ===\n");
        if (target == null) {
            out.append("target is null\n");
            return out.toString();
        }

        Field field = findField(target.getClass(), fieldName);
        if (field == null) {
            out.append("no field named ").append(fieldName).append(" on ")
                    .append(target.getClass().getName()).append('\n');
            return out.toString();
        }

        Object value;
        try {
            field.setAccessible(true);
            value = field.get(Modifier.isStatic(field.getModifiers()) ? null : target);
        } catch (ReflectiveOperationException | RuntimeException e) {
            out.append("unreadable: ").append(e.getClass().getSimpleName()).append('\n');
            return out.toString();
        }

        if (value == null) {
            out.append("field is null\n");
            return out.toString();
        }
        if (!(value instanceof java.util.Collection)) {
            out.append("not a collection: ").append(summarise(value)).append('\n');
            return out.toString();
        }

        int index = 0;
        for (Object element : (java.util.Collection<?>) value) {
            out.append('[').append(index++).append("] ");
            if (element == null) {
                out.append("null\n");
                continue;
            }
            out.append(element.getClass().getName());
            if (isPlatformClass(element.getClass())) {
                // Platform internals are never the subject here, and reflecting
                // into them trips the module system's illegal-access warning
                // today and a hard denial in a later JDK.
                out.append(" = ").append(summarise(element)).append('\n');
                continue;
            }
            out.append('\n');
            // Walk the hierarchy, not just the element's own class. Engine
            // entity types are deep -- units.al extends v extends am -- and the
            // state that identifies an object (owner, position, health) is
            // declared on the base. Listing only declared fields makes a
            // subclass look featureless, which is precisely how a tree class
            // was once mistaken for the unit class.
            for (Class<?> type : ownedHierarchy(element.getClass())) {
                for (Field own : declaredFieldsOf(type)) {
                    if (Modifier.isStatic(own.getModifiers())) {
                        continue;
                    }
                    out.append("      ").append(describeField(element, own)).append('\n');
                }
            }
        }
        return out.toString();
    }

    /**
     * Searches the live object graph for collections holding a given kind of element.
     *
     * <p>Written after guessing failed. Correlating obfuscated field types by
     * hand produced two confident wrong answers -- a sprite registry read as
     * the unit list because its size coincided, and a graph node read as the
     * unit class because its query methods looked plausible. Asking the running
     * game which collections actually hold {@code game.units.*} objects removes
     * the guessing step entirely.
     *
     * <p>Breadth-first and bounded on both depth and node count, because the
     * engine graph is cyclic and large. Identity-based visitation, since engine
     * classes override neither {@code equals} nor {@code hashCode} predictably
     * and calling either would run engine code from a probe thread.
     *
     * @param root Object to search from.
     * @param elementPrefix Binary-name prefix an element class must start with.
     * @param maxDepth Maximum field hops from the root.
     * @param maxNodes Maximum objects to visit before stopping.
     * @return A report naming the field path, size and element class of each hit.
     */
    static String findCollections(Object root, String elementPrefix, int maxDepth, int maxNodes) {
        StringBuilder out = new StringBuilder();
        out.append("=== collections of ").append(elementPrefix)
                .append(" (depth<=").append(maxDepth).append(") ===\n");
        if (root == null) {
            out.append("root is null\n");
            return out.toString();
        }

        java.util.Map<Object, Boolean> seen = new java.util.IdentityHashMap<Object, Boolean>();
        java.util.ArrayDeque<Node> queue = new java.util.ArrayDeque<Node>();
        queue.add(new Node(root, "", 0));
        seen.put(root, Boolean.TRUE);
        int visited = 0;
        int hits = 0;

        while (!queue.isEmpty() && visited < maxNodes) {
            Node node = queue.poll();
            visited++;
            if (node.depth >= maxDepth) {
                continue;
            }
            // A container is traversed by its ELEMENTS, never by its declared
            // fields. Reflecting into java.util internals (ArrayList.elementData,
            // AbstractList.modCount, Arrays$ArrayList.a) reaches the same objects
            // by a route the module system warns about today and denies in a
            // later JDK -- and it is route-dependent besides, since each
            // Collection implementation stores its contents differently.
            java.util.Collection<?> contained = containedValues(node.value);
            if (contained != null) {
                int index = 0;
                for (Object element : contained) {
                    String path = node.path + "[" + index + "]";
                    index++;
                    hits += consider(out, element, path, node, seen, queue, elementPrefix);
                }
                continue;
            }
            // Any other platform object is a leaf: its internals are never the
            // subject of a game-object search.
            if (isPlatformClass(node.value.getClass())) {
                continue;
            }
            for (Class<?> type : ownedHierarchy(node.value.getClass())) {
                for (Field field : declaredFieldsOf(type)) {
                    Object child = readQuietly(node.value, field);
                    String path = node.path + "." + field.getName();
                    hits += consider(out, child, path, node, seen, queue, elementPrefix);
                }
            }
        }
        out.append("visited ").append(visited).append(" objects, ")
                .append(hits).append(" hit(s)\n");
        return out.toString();
    }

    /**
     * Returns the values a container holds, or null when the object is not one.
     *
     * <p>Arrays are included: {@code java.lang.reflect.Array} reads them through
     * a supported API rather than through their fields, which they do not have.
     *
     * @param value The object to inspect.
     * @return Its contained values, or null when it is not a container.
     */
    private static java.util.Collection<?> containedValues(Object value) {
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

    /**
     * Records a reached object as a hit when it matches, and queues it for descent.
     *
     * <p>Shared by both traversal routes -- field walk and container expansion --
     * so a match is recognised identically however it was reached.
     *
     * @param out Report being accumulated.
     * @param child The object reached, possibly null.
     * @param path Path expression that reached it.
     * @param node The node it was reached from.
     * @param seen Identity set of already-visited objects.
     * @param queue Traversal queue to extend.
     * @param elementPrefix Class-name prefix that makes a container a hit.
     * @return 1 when the object was reported as a hit, 0 otherwise.
     */
    private static int consider(
            StringBuilder out,
            Object child,
            String path,
            Node node,
            java.util.Map<Object, Boolean> seen,
            java.util.ArrayDeque<Node> queue,
            String elementPrefix) {
        if (child == null || seen.containsKey(child)) {
            return 0;
        }
        seen.put(child, Boolean.TRUE);
        int hits = 0;
        String elementClass = elementClassOf(child);
        if (elementClass != null && elementClass.startsWith(elementPrefix)) {
            out.append(path).append("  ").append(summarise(child)).append('\n');
            hits = 1;
        }
        if (!isPlatformClass(child.getClass()) || containedValues(child) != null) {
            queue.add(new Node(child, path, node.depth + 1));
        }
        return hits;
    }

    /** Returns the class of a collection's or array's first non-null element. */
    private static String elementClassOf(Object value) {
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
    private static Object readQuietly(Object owner, Field field) {
        try {
            field.setAccessible(true);
            return field.get(Modifier.isStatic(field.getModifiers()) ? null : owner);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** One queued object with the field path that reached it. */
    private static final class Node {
        final Object value;
        final String path;
        final int depth;

        Node(Object value, String path, int depth) {
            this.value = value;
            this.path = path;
            this.depth = depth;
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
    private static boolean isPlatformClass(Class<?> type) {
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
    private static Field findField(Class<?> type, String name) {
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
    private static java.util.List<Class<?>> ownedHierarchy(Class<?> type) {
        java.util.List<Class<?>> chain = new java.util.ArrayList<Class<?>>();
        for (Class<?> current = type;
                current != null && current != Object.class && !isPlatformClass(current);
                current = current.getSuperclass()) {
            chain.add(current);
        }
        return chain;
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
