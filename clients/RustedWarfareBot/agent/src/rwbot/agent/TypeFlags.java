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

        java.util.LinkedHashMap<String, Combat> reach = new java.util.LinkedHashMap<String, Combat>();
        for (Object type : allTypes()) {
            reach.put(Perception.nameOfType(type), combatOf(type));
        }

        StringBuilder out = new StringBuilder();
        int index = 0;
        for (java.util.Map.Entry<String, Boolean> entry : flags.entrySet()) {
            out.append(record(index++, entry.getKey(), entry.getValue().booleanValue()))
                    .append('\n');
        }
        index = 0;
        for (java.util.Map.Entry<String, Combat> entry : reach.entrySet()) {
            out.append(combatRecord(index++, entry.getKey(), entry.getValue())).append('\n');
        }
        return out.toString();
    }

    /**
     * One type's combat facts, as its prototype answers them.
     *
     * <p>Range and reachable layers travel together because they are read in one
     * pass over one prototype and are meaningless apart: a range with no layers
     * describes a unit that can shoot 130 units at nothing.
     */
    static final class Combat {

        /** The unarmed answer: no range, and no layer it can reach. */
        static final Combat UNARMED = new Combat(0.0f, false, false, false, false);

        private final float attackRange;
        private final boolean hitsLand;
        private final boolean hitsAir;
        private final boolean hitsUnderwater;
        private final boolean hitsLandOutOfWater;

        Combat(
                float attackRange,
                boolean hitsLand,
                boolean hitsAir,
                boolean hitsUnderwater,
                boolean hitsLandOutOfWater) {
            this.attackRange = attackRange;
            this.hitsLand = hitsLand;
            this.hitsAir = hitsAir;
            this.hitsUnderwater = hitsUnderwater;
            this.hitsLandOutOfWater = hitsLandOutOfWater;
        }

        float attackRange() {
            return this.attackRange;
        }

        boolean hitsLand() {
            return this.hitsLand;
        }

        boolean hitsAir() {
            return this.hitsAir;
        }

        boolean hitsUnderwater() {
            return this.hitsUnderwater;
        }

        boolean hitsLandOutOfWater() {
            return this.hitsLandOutOfWater;
        }
    }

    /**
     * Returns a type's range and the layers it can shoot onto.
     *
     * <p>Read off the engine's prototype for the type rather than off a live
     * unit, so asking costs a map lookup and spawns nothing.
     *
     * <p>Unarmed types are reported as zero range and no reachable layer rather
     * than omitted. "This type cannot shoot" is an answer the threat model
     * needs, and a missing record would be indistinguishable from a type the
     * dump never reached — which is exactly the ambiguity that made the stat
     * catalogue unsafe to read reach from (wiki: policy-threat).
     *
     * <p><b>The layer predicates are forced to false for the unarmed rather
     * than reported as the engine answers them.</b> Their base implementations
     * return true for air and land regardless of armament, because the engine
     * only ever consults them after establishing that a weapon exists. Reporting
     * that unfiltered would put "a Builder can shoot aircraft" on the wire.
     *
     * @param type The unit type.
     * @return Its combat facts, or {@link Combat#UNARMED}.
     */
    private static Combat combatOf(Object type) {
        Object prototype =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(
                                EngineAccess.pinnedClass(EngineNames.ENTITY_CLASS),
                                TypeNames.TYPE_PROTOTYPE,
                                EngineAccess.pinnedClass(TypeNames.TYPE_CLASS)),
                        null,
                        type);
        // A type with no prototype, and a prototype that takes no orders, are
        // both unarmed for this purpose: neither can appear as a hostile
        // holding a weapon, which is the only question being asked.
        if (prototype == null
                || !EngineAccess.pinnedClass(EngineNames.ORDERABLE_CLASS).isInstance(prototype)) {
            return Combat.UNARMED;
        }
        Object armed =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(prototype.getClass(), TypeNames.UNIT_ARMED),
                        prototype);
        if (!Boolean.TRUE.equals(armed)) {
            return Combat.UNARMED;
        }
        Object range =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(
                                prototype.getClass(), TypeNames.UNIT_ATTACK_RANGE),
                        prototype);
        if (!(range instanceof Float)) {
            throw new IllegalStateException(
                    "rw-agent: " + TypeNames.UNIT_ATTACK_RANGE + "() did not return a float"
                            + EngineNames.PIN);
        }
        return new Combat(
                ((Float) range).floatValue(),
                layerPredicate(prototype, TypeNames.UNIT_HITS_LAND),
                layerPredicate(prototype, TypeNames.UNIT_HITS_AIR),
                layerPredicate(prototype, TypeNames.UNIT_HITS_UNDERWATER),
                layerPredicate(prototype, TypeNames.UNIT_HITS_LAND_OUT_OF_WATER));
    }

    /**
     * Asks a prototype one of its layer-attack predicates.
     *
     * @param prototype The type's prototype entity.
     * @param name Pinned accessor name.
     * @return The prototype's own answer.
     * @throws IllegalStateException When the accessor does not return a boolean,
     *     which is a pinned name that has moved rather than a unit that cannot
     *     answer.
     */
    private static boolean layerPredicate(Object prototype, String name) {
        Object answer =
                EngineAccess.invoke(
                        EngineAccess.pinnedMethod(prototype.getClass(), name), prototype);
        if (!(answer instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + name + "() on " + prototype.getClass().getName()
                            + " did not return a boolean" + EngineNames.PIN);
        }
        return ((Boolean) answer).booleanValue();
    }

    /**
     * Renders one type's combat facts.
     *
     * <p>Its own record kind rather than another field on the placement record,
     * because the two answer unrelated questions and a type named for placement
     * carrying an attack range would be a lie the decoder then has to keep
     * telling. The file already carries more than one kind.
     */
    static String combatRecord(int index, String name, Combat combat) {
        StringBuilder out = new StringBuilder();
        out.append("{\"kind\":\"unitcombat\",\"index\":").append(index).append(",\"name\":");
        Json.quote(out, name);
        out.append(",\"attack_range\":").append(combat.attackRange());
        out.append(",\"hits_land\":").append(combat.hitsLand());
        out.append(",\"hits_air\":").append(combat.hitsAir());
        out.append(",\"hits_underwater\":").append(combat.hitsUnderwater());
        out.append(",\"hits_land_out_of_water\":").append(combat.hitsLandOutOfWater());
        out.append('}');
        return out.toString();
    }

    /**
     * Returns every registered unit type, built-in and asset-defined alike.
     *
     * <p>Shared with {@link BuildTree}, which asks the same set a different
     * question. Enumeration is the fiddly half -- two registries, one enum and
     * one static list, with asset units deliberately shadowing built-ins of the
     * same name -- so it lives in one place and both readers get the same set.
     *
     * @return The types, built-ins first.
     */
    static java.util.List<Object> allTypes() {
        java.util.List<Object> types = new java.util.ArrayList<Object>();
        for (Object type : builtInTypes()) {
            types.add(type);
        }
        for (Object type : assetTypes()) {
            types.add(type);
        }
        return types;
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
        Class<?> registry = EngineAccess.pinnedClass(TypeNames.TYPE_REGISTRY_CLASS);
        Object[] constants = registry.getEnumConstants();
        if (constants == null) {
            throw new IllegalStateException(
                    "rw-agent: " + TypeNames.TYPE_REGISTRY_CLASS
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
        Class<?> custom = EngineAccess.pinnedClass(TypeNames.CUSTOM_TYPE_CLASS);
        Object value;
        try {
            value = EngineAccess.pinnedField(custom, TypeNames.CUSTOM_TYPE_LIST).get(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "rw-agent: cannot read " + TypeNames.CUSTOM_TYPE_LIST
                            + EngineNames.PIN, e);
        }
        if (!(value instanceof Iterable)) {
            throw new IllegalStateException(
                    "rw-agent: " + TypeNames.CUSTOM_TYPE_CLASS + "."
                            + TypeNames.CUSTOM_TYPE_LIST + " is not iterable"
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
                                type.getClass(), TypeNames.TYPE_NEEDS_POOL),
                        type);
        if (!(answer instanceof Boolean)) {
            throw new IllegalStateException(
                    "rw-agent: " + TypeNames.TYPE_NEEDS_POOL + "() on "
                            + type.getClass().getName() + " did not return a boolean"
                            + EngineNames.PIN);
        }
        return ((Boolean) answer).booleanValue();
    }

}
