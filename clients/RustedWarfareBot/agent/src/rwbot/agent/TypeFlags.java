package rwbot.agent;

/**
 * Which unit types the engine will only place on a resource pool.
 *
 * <p>The planner has to know this before it chooses a site, and nothing it
 * already reads will tell it. The unit catalogue from {@code -printunits}
 * carries prices and stats but no placement rules; the only mention of the rule
 * anywhere in the shipped files is an English sentence in a translations
 * bundle, which is a blurb rather than a fact (wiki: mechanics-resource-pools).
 *
 * <p>So it is read from the live engine. Every type name is resolved through the
 * engine's own by-name lookup — the same call the build path makes — and asked
 * for its placement predicate. What this dumps is therefore not a transcription
 * of the rule but the rule itself, answered by the object that will later
 * enforce it.
 *
 * <p>Output is NDJSON, one flat object per type, for the same reason the world
 * stream is (see {@link StateStream}): the consumer is strictly typed and
 * cannot use a lenient parser.
 */
final class TypeFlags {

    private TypeFlags() {
    }

    /**
     * Renders every registered unit type and its resource-pool requirement.
     *
     * <p>Both kinds of type are enumerated. Built-ins are enum constants;
     * everything under {@code assets/units/} is loaded into a separate list.
     * Names collide between the two on purpose — an asset unit declaring
     * {@code overrideAndReplace} deliberately shadows a built-in of the same
     * name — so the asset pass runs second and overwrites, which is the
     * engine's own precedence: its by-name lookup consults the asset units
     * before the built-in enum.
     *
     * <p>Each type is asked directly rather than looked back up by the name it
     * just reported, because that round trip does not always close. One
     * built-in type reports the readable name {@code marker} while the registry
     * matches built-ins on their enum constant name, so resolving
     * {@code marker} returns nothing at all. Asking the object in hand avoids
     * inventing an answer for a discrepancy that belongs to the engine.
     *
     * <p>Must run on the game thread: the asset-defined types are loaded
     * during map load.
     *
     * @return One newline-terminated record per distinct type name.
     * @throws IllegalStateException When the built-in registry cannot be
     *     enumerated, or a type does not answer the placement predicate.
     */
    static String dump() {
        java.util.LinkedHashMap<String, Boolean> flags =
                new java.util.LinkedHashMap<String, Boolean>();
        for (Object type : builtInTypes()) {
            flags.put(Perception.nameOfType(type), Boolean.valueOf(needsPool(type)));
        }
        for (Object type : assetTypes()) {
            flags.put(Perception.nameOfType(type), Boolean.valueOf(needsPool(type)));
        }

        StringBuilder out = new StringBuilder();
        int index = 0;
        for (java.util.Map.Entry<String, Boolean> entry : flags.entrySet()) {
            out.append(record(index++, entry.getKey(), entry.getValue().booleanValue()))
                    .append('\n');
        }
        return out.toString();
    }

    /** Renders one type record. */
    static String record(int index, String name, boolean needsPool) {
        StringBuilder out = new StringBuilder();
        out.append("{\"kind\":\"unittype\",\"index\":").append(index).append(",\"name\":");
        Json.quote(out, name);
        out.append(",\"needs_pool\":").append(needsPool).append('}');
        return out.toString();
    }

    /**
     * Returns the built-in unit types.
     *
     * <p>The registry is an enum, so its constants are the whole set. A null
     * answer means it stopped being one, which is a pinned-name failure rather
     * than an empty catalogue, and is reported as such.
     *
     * @return The enum constants.
     */
    private static Object[] builtInTypes() {
        Class<?> registry = EngineAccess.pinnedClass(EngineNames.TYPE_REGISTRY_CLASS);
        Object[] constants = registry.getEnumConstants();
        if (constants == null) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TYPE_REGISTRY_CLASS
                            + " is no longer an enum, so its built-in types cannot be"
                            + " enumerated" + EngineNames.PIN);
        }
        return constants;
    }

    /**
     * Returns the asset-defined unit types.
     *
     * @return Every type loaded from {@code assets/units/} and from any enabled
     *     mod.
     */
    private static Iterable<?> assetTypes() {
        Class<?> custom = EngineAccess.pinnedClass(EngineNames.CUSTOM_TYPE_CLASS);
        Object value;
        try {
            value = EngineAccess.pinnedField(custom, EngineNames.CUSTOM_TYPE_LIST).get(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read " + EngineNames.CUSTOM_TYPE_LIST
                            + EngineNames.PIN, e);
        }
        if (!(value instanceof Iterable)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.CUSTOM_TYPE_CLASS + "."
                            + EngineNames.CUSTOM_TYPE_LIST + " is not iterable"
                            + EngineNames.PIN);
        }
        return (Iterable<?>) value;
    }

    /**
     * Asks a type whether it may only be placed on a resource pool.
     *
     * @param type The unit type.
     * @return The type's own answer.
     * @throws IllegalStateException When the predicate does not answer.
     */
    private static boolean needsPool(Object type) {
        Object answer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(
                                type.getClass(), EngineNames.TYPE_NEEDS_POOL),
                        type);
        if (!(answer instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + EngineNames.TYPE_NEEDS_POOL + "() on "
                            + type.getClass().getName() + " did not return a boolean"
                            + EngineNames.PIN);
        }
        return ((Boolean) answer).booleanValue();
    }

}
