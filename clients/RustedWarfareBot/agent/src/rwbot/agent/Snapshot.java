package rwbot.agent;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Reflective snapshots of one engine object.
 *
 * <p>{@code game-lib.jar} is obfuscated: the engine's base class carries some
 * 360 members of which exactly one kept a readable name, so identifying a field
 * by reading bytecode means correlating hundreds of single-letter names by type.
 * The running object graph is the cheaper source of truth — a list whose size
 * matches a count the engine already printed identifies itself.
 *
 * <p>Two depths. {@link #describe} lists an object's own fields;
 * {@link #describeElements} expands one named collection a further level,
 * because a snapshot can say a collection holds eleven things but not what
 * distinguishes them. Walking the whole graph instead is {@link GraphSearch}.
 *
 * <p>Reads only. The rendering rules are {@link ObjectView}'s.
 */
final class Snapshot {

    private Snapshot() {
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

        for (Class<?> type : ObjectView.ownedHierarchy(target.getClass())) {
            out.append("--- ").append(type.getName()).append(" ---\n");
            for (Field field : ObjectView.declaredFieldsOf(type)) {
                out.append(ObjectView.describeField(target, field)).append('\n');
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

        Field field = ObjectView.findField(target.getClass(), fieldName);
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
            out.append("not a collection: ").append(ObjectView.summarise(value)).append('\n');
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
            if (ObjectView.isPlatformClass(element.getClass())) {
                // Platform internals are never the subject here, and reflecting
                // into them trips the module system's illegal-access warning
                // today and a hard denial in a later JDK.
                out.append(" = ").append(ObjectView.summarise(element)).append('\n');
                continue;
            }
            out.append('\n');
            // Walk the hierarchy, not just the element's own class. Engine
            // entity types are deep -- units.al extends v extends am -- and the
            // state that identifies an object (owner, position, health) is
            // declared on the base. Listing only declared fields makes a
            // subclass look featureless, which is precisely how a tree class
            // was once mistaken for the unit class.
            for (Class<?> type : ObjectView.ownedHierarchy(element.getClass())) {
                for (Field own : ObjectView.declaredFieldsOf(type)) {
                    if (Modifier.isStatic(own.getModifiers())) {
                        continue;
                    }
                    out.append("      ").append(ObjectView.describeField(element, own)).append('\n');
                }
            }
        }
        return out.toString();
    }
}
