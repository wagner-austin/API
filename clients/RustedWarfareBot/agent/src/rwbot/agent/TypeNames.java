package rwbot.agent;

/**
 * Every obfuscated name that describes a unit <i>type</i>, and nothing else.
 *
 * <p>Pure data, and the same discipline as {@link EngineNames}, which this was
 * split out of when that table reached 674 lines. The seam is the subject: a
 * name here answers what a type is, what it can be told to make, and what it
 * can shoot. Names describing an entity standing on the map, a player, or the
 * map itself stay next door.
 *
 * <p>Kept apart because the two grow for different reasons. A new thing the bot
 * learns to <i>read</i> adds a name to {@link EngineNames}; a new thing it
 * learns to <i>build</i> adds one here, and the build tree is the half that has
 * grown fastest.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> Every name
 * below moves between releases, silently -- the jar is obfuscated, so a rename
 * produces a binding that resolves to nothing rather than a compile error.
 * {@link BindingCheck#verifyBindings()} resolves all of them against the jar
 * with no game running, and {@code make check} runs it, so a game update fails
 * at the gate rather than in a live run.
 *
 * <p>Each name carries why it is trusted, not just what it is. A name derived
 * from the engine printing it in its own log is a different kind of fact from
 * one inferred by matching a field's shape, and the comment is where that
 * distinction survives.
 */
final class TypeNames {

    private TypeNames() {
    }

    /** The unit-type interface a placement command carries. */
    static final String TYPE_CLASS = "com.corrodinggames.rts.game.units.as";

    /** Holds the by-name unit-type lookup. Also the built-in unit-type enum. */
    static final String TYPE_REGISTRY_CLASS = "com.corrodinggames.rts.game.units.ar";

    /**
     * An asset-defined unit type, as opposed to a built-in enum constant.
     *
     * <p>Most of what the game ships is defined this way: {@code assets/units/}
     * holds an {@code .ini} per unit and the loader turns each into one of
     * these. Both kinds implement {@link #TYPE_CLASS}, so once enumerated they
     * are read identically.
     */
    static final String CUSTOM_TYPE_CLASS = "com.corrodinggames.rts.game.units.custom.l";

    /** Static list of every asset-defined unit type. */
    static final String CUSTOM_TYPE_LIST = "d";

    /** Entity accessor returning its unit type. */
    static final String TYPE_ACCESSOR = "r";

    /** Unit-type accessor returning its readable name. */
    static final String TYPE_NAME_ACCESSOR = "i";

    /**
     * Unit-type predicate: true when the type may only be placed on a resource
     * pool.
     *
     * <p>This is the engine's own placement rule, not a reading of the unit's
     * blurb. The chain is: an {@code .ini} declares
     * {@code placeOnlyOnResPool: true}, the loader stores it, this accessor
     * reports it, and the placement check consults it before anything else
     * (wiki: mechanics-resource-pools).
     */
    static final String TYPE_NEEDS_POOL = "p";

    /**
     * Static lookup from a unit type to its prototype entity.
     *
     * <p>A map read, not a construction: the engine keeps one prototype per
     * registered type and hands the same object back. Nothing is spawned and
     * nothing is registered, which is what makes it safe to ask 173 types for
     * their stats mid-game. The engine's own {@code -printunits} reads exactly
     * this way (wiki: mechanics-unit-catalogue).
     */
    static final String TYPE_PROTOTYPE = "a";

    /**
     * Unit-type accessor returning its action list at a tech level.
     *
     * <p>The static counterpart of {@link #ACTIONS}. That one is read off a
     * live entity and answers "what can this unit make now"; this one is asked
     * of a type in the registry and answers "what would one of these make",
     * which is the question a plan has to answer before the thing exists.
     *
     * <p>Declared on the type interface itself, so every registered type
     * answers it -- the per-class overrides of {@link #ACTIONS} all delegate
     * here, which is why the two never disagree.
     */
    static final String TYPE_BUILD_ACTIONS = "a";

    /**
     * One thing a unit can do: an action, in the engine's own sense.
     *
     * <p>Building a structure and producing a unit are the same mechanism here.
     * A builder's actions yield buildings and a factory's yield units, and the
     * order path resolves either through the same lookup, which is why one
     * command verb covers both (wiki: mechanics-build-actions).
     */
    static final String ACTION_CLASS = "com.corrodinggames.rts.game.units.a.s";

    /**
     * Entity accessor returning its action list.
     *
     * <p>Declared on the entity base and overridden per unit class, so the
     * answer belongs to the class rather than to the type registry — which is
     * why the options are read off live entities rather than dumped from the
     * registry the way placement flags are.
     */
    static final String ACTIONS = "N";

    /**
     * Action accessor returning the type it makes.
     *
     * <p>Abstract on the action base, so every action answers it. This is the
     * general "what does this action concern"; the narrower
     * {@link #ACTION_PLACED_TYPE} is set only on the ones that place something
     * at a position.
     */
    static final String ACTION_MAKES = "i";

    /**
     * Action accessor returning the type it places at a position, or null.
     *
     * <p>The discriminator between the two dispatch verbs, and it is the
     * engine's own: the build-waypoint lookup matches candidate actions on
     * exactly this accessor, so an action for which it is null cannot be
     * reached that way and must be dispatched by action key instead
     * (wiki: mechanics-build-actions).
     */
    static final String ACTION_PLACED_TYPE = "y";

    /** Action predicate: whether it makes something at all, as opposed to a stop or a rally. */
    static final String ACTION_MAKES_SOMETHING = "g";

    /**
     * Action accessor returning its interned key.
     *
     * <p>What an action command carries. Read off the action rather than built
     * from a string, so the agent never has to know the engine's key format.
     */
    static final String ACTION_KEY = "N";

    /** The interned identifier an action command carries. */
    static final String ACTION_KEY_CLASS = "com.corrodinggames.rts.game.units.a.c";

    /** Accessor on that identifier returning its readable name. */
    static final String ACTION_KEY_NAME = "a";

    /**
     * Action accessor returning its selector index.
     *
     * <p>The same integer a build command carries. Passing -1 means "whichever
     * action produces this type"; the engine's own lookup compares against this
     * (wiki: building-structures).
     */
    static final String ACTION_INDEX = "t";

    /** Action predicate: whether the given unit may use it right now. */
    static final String ACTION_AVAILABLE = "b";

    /**
     * Action accessor: what the action costs in credits.
     *
     * <p>Abstract on the action base class, so every action answers it. A
     * produce action returns its unit's price; a factory's tier upgrade
     * returns the tier price; a rally or stop returns zero. That last
     * contrast is what the tech channel selects on -- the upgrade and the
     * rally are otherwise identical on the wire, both concerning no type,
     * and the first live probe spent four unlock budgets setting rally
     * points (decompiled {@code units/a/o.java} vs {@code units/d/n.java},
     * wiki: mechanics-build-actions).
     */
    static final String ACTION_PRICE = "c";

    /** Action predicate: whether the given unit has it locked. */
    static final String ACTION_LOCKED = "g";

    /**
     * Action predicate: whether it applies to the given unit at all.
     *
     * <p>Broader than it looks. Its body folds in {@link #ACTION_LOCKED}, a
     * per-key cooldown, and an affordability test, so a false answer has four
     * possible causes and the narrower predicates are what separate them.
     *
     * <p><b>The boolean argument must be false.</b> Passed true, the engine
     * routes the affordability test through its check-and-charge helper, which
     * deducts the cost on success. The predicate is only a predicate on the
     * false branch; on the true branch it is a purchase. That is why every
     * caller here pins the argument rather than threading it through
     * (wiki: mechanics-build-actions).
     */
    static final String ACTION_APPLIES = "a";

    /**
     * Production queue on a building that makes units.
     *
     * <p>Present only on producing buildings, which is itself the fact: an
     * entity without the field makes nothing and queues nothing. Absence is
     * therefore read as depth zero rather than treated as an error.
     */
    static final String PRODUCTION_QUEUE = "z";

    /** The queued items inside that queue. An {@code AbstractList}. */
    static final String QUEUE_ITEMS = "c";

    /**
     * The production-queue class.
     *
     * <p>Needed because the field name alone does not identify the queue.
     * Obfuscation reuses single letters freely, and other classes carry an
     * unrelated field of the same name — one of them crashed a run when it was
     * read as a queue. The field is matched by declared type as well as name.
     */
    static final String QUEUE_CLASS = "com.corrodinggames.rts.game.units.d.k";

    /** The point an action command carries. */
    static final String POINT_CLASS = "android.graphics.PointF";

    /**
     * Entity predicate: whether it has a weapon at all.
     *
     * <p>Abstract on the entity base, so every unit answers it. The same gate
     * {@code -printunits} applies before printing an attack range.
     */
    static final String UNIT_ARMED = "l";

    /**
     * Orderable accessor returning its attack range in world units.
     *
     * <p>The figure {@code -printunits} labels "Attack Range", read from the
     * accessor rather than from that output — which is the point, since the
     * output covers 90 of 173 registered types and the threat model needs all
     * of them (wiki: policy-threat).
     */
    static final String UNIT_ATTACK_RANGE = "m";

    /**
     * Orderable predicate: whether it can shoot a target on the water layer.
     *
     * <p>These three are the accessors the engine's own attackability test
     * consults, rather than a model of it. That test reads, in the decompile,
     * {@code if (target.isFlying()) return canAttackAir(); if
     * (target.isUnderwater()) return canAttackUnderwater(); ... return
     * canAttackLand();} — so asking the same three accessors per type answers
     * the same question the engine will answer at fire time, one dispatch
     * earlier.
     *
     * <p>The mapping is confirmed through the asset loader rather than guessed
     * from the accessor order: the custom-unit override returns the
     * {@code canAttackUnderwaterUnits}, {@code canAttackFlyingUnits} and
     * {@code canAttackLandUnits} predicates respectively, which are the three
     * keys an {@code .ini} declares in its {@code [attack]} section.
     *
     * <p><b>Read on the prototype, so a dynamic condition is not honoured.</b>
     * A unit may declare {@code canAttackCondition} as a logic expression
     * evaluated against the live unit; asking the prototype evaluates it against
     * the prototype instead. Nothing in the base game's buildable set does that,
     * and the alternative — asking per attacker-target pair every sample — is a
     * reflective call per pair per tick for an answer that does not change.
     */
    static final String UNIT_HITS_UNDERWATER = "ae";

    /** Orderable predicate: whether it can shoot a target on the air layer. */
    static final String UNIT_HITS_AIR = "af";

    /** Orderable predicate: whether it can shoot a target on the ground. */
    static final String UNIT_HITS_LAND = "ag";

    /**
     * Orderable predicate: whether its ground fire reaches targets clear of water.
     *
     * <p>The {@code canAttackNotTouchingWaterUnits} key, and the one branch of
     * the engine's test that is neither air nor underwater: a weapon declaring
     * this false — a torpedo, in practice — hits a ground target only while that
     * target is in the water. True for everything the base game lets a player
     * build, which is exactly why it is carried rather than assumed: an
     * assumption that holds for every unit tested so far is the kind that fails
     * first on a water map.
     */
    static final String UNIT_HITS_LAND_OUT_OF_WATER = "ah";
}
