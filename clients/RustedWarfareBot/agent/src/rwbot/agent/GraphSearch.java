package rwbot.agent;

import java.lang.reflect.Field;

/**
 * A bounded walk over the live object graph, looking for collections of a
 * given kind of element.
 *
 * <p>Written after guessing failed. Correlating obfuscated field types by hand
 * produced two confident wrong answers — a sprite registry read as the unit
 * list because its size coincided, and a graph node read as the unit class
 * because its query methods looked plausible. Asking the running game which
 * collections actually hold a given type removes the guessing step entirely.
 *
 * <p>Breadth-first and bounded on both depth and node count, because the engine
 * graph is cyclic and large. Visitation is identity-based, since engine classes
 * override neither {@code equals} nor {@code hashCode} predictably and calling
 * either would run engine code from a probe thread.
 *
 * <p><b>A container is traversed by its elements, never by its declared
 * fields.</b> Reflecting into {@code java.util} internals reaches the same
 * objects by a route the module system denies in later JDKs, and one that is
 * implementation-specific besides. Looking at a single object instead is
 * {@link Snapshot}; the rendering rules are {@link ObjectView}'s.
 */
final class GraphSearch {

    private GraphSearch() {
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
            java.util.Collection<?> contained = ObjectView.containedValues(node.value);
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
            if (ObjectView.isPlatformClass(node.value.getClass())) {
                continue;
            }
            for (Class<?> type : ObjectView.ownedHierarchy(node.value.getClass())) {
                for (Field field : ObjectView.declaredFieldsOf(type)) {
                    Object child = ObjectView.readQuietly(node.value, field);
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
        String elementClass = ObjectView.elementClassOf(child);
        if (elementClass != null && elementClass.startsWith(elementPrefix)) {
            out.append(path).append("  ").append(ObjectView.summarise(child)).append('\n');
            hits = 1;
        }
        if (!ObjectView.isPlatformClass(child.getClass()) || ObjectView.containedValues(child) != null) {
            queue.add(new Node(child, path, node.depth + 1));
        }
        return hits;
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
}
