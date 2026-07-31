package rwbot.agent;

/**
 * Every obfuscated name the agent depends on, and nothing else.
 *
 * <p>Pure data. This file exists separately because it is the part that grows:
 * each new thing the bot learns to read adds a name here while the reflection
 * that reaches it and the guard that checks it stay exactly as they were.
 * Keeping the table apart means that growth lands somewhere with no logic in
 * it.
 *
 * <p><b>Pinned to Rusted Warfare 1.15 (code 176, build #28).</b> Every name
 * below moves between releases, silently — the jar is obfuscated, so a rename
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
final class EngineNames {

    private EngineNames() {
    }

    static final String ENTITY_CLASS = "com.corrodinggames.rts.game.units.am";

    static final String TREE_CLASS = "com.corrodinggames.rts.game.units.al";

    static final String ORDERABLE_CLASS = "com.corrodinggames.rts.game.units.y";

    static final String TEAM_CLASS = "com.corrodinggames.rts.game.n";

    static final String COMMAND_CLASS = "com.corrodinggames.rts.gameFramework.e";

    static final String CONTROLLER_CLASS = "com.corrodinggames.rts.gameFramework.c";

    static final String SCRIPTS_CLASS = "com.corrodinggames.librocket.scripts.ScriptEngine";

    /**
     * Starts a match with a chosen map, opponent count and team split.
     *
     * <p>What {@code Root.loadConfigCommon} ends at, after reading every figure
     * out of a GUI document. Calling it directly is what lets the map and the
     * opponent count be experiment variables rather than Java fallbacks
     * (wiki: policy-determinism).
     */
    static final String MATCH_STARTER_CLASS = "com.corrodinggames.rts.appFramework.i";

    /** {@code (String map, boolean skirmish, int ais, int allies, boolean, boolean)}. */
    static final String MATCH_START_METHOD = "a";

    /** The settings object on the engine, holding {@link #AI_DIFFICULTY_FIELD}. */
    static final String SETTINGS_FIELD = "bQ";

    /** Queues one script line for the engine to run on its own thread. */
    static final String SCRIPT_QUEUE_METHOD = "addScriptToQueue";

    /**
     * AI difficulty, -2 to 3.
     *
     * <p>Not obfuscated, because {@code SettingsEngine} is not. It is an income
     * multiplier applied to AI players only -- 0.4x, 0.7x, 1.0x, 1.4x, 1.8x and
     * 3.7x across the scale -- so at the default of 0 an opponent earns exactly
     * what the bot does.
     */
    static final String AI_DIFFICULTY_FIELD = "aiDifficulty";

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

    /** The map: terrain grid, tile layers, and the per-tile fog test. */
    static final String MAP_CLASS = "com.corrodinggames.rts.game.b.b";

    /** One map tile, carrying the flags its tileset declared. */
    static final String TILE_CLASS = "com.corrodinggames.rts.game.b.g";

    /** Static master entity list on the entity base class; holds units and trees alike. */
    static final String ENTITY_LIST = "bE";

    /** Owning player on an entity. */
    static final String OWNER = "bX";

    /** World position on an entity. */
    static final String POS_X = "eo";

    static final String POS_Y = "ep";

    /** The engine's current player. */
    static final String LOCAL_TEAM = "bs";

    /** Team number on a player, as the engine prints it in its own AI warnings. */
    static final String TEAM_ID = "k";

    /**
     * Team predicate: whether another player is hostile to this one.
     *
     * <p>Not the negation of "mine", and the difference is not academic. The
     * engine compares alliance group rather than team number, and answers false
     * whenever either side is the neutral team — so a shared-alliance ally and a
     * neutral map object both read as not hostile, which "everything that is not
     * mine" gets wrong in both directions.
     *
     * <p>Its sibling {@code d(n)} is the same comparison inverted, which is what
     * pins this one as the hostile direction rather than the friendly one.
     */
    static final String TEAM_HOSTILE_TO = "c";

    /**
     * Units a player holds, excluding buildings and including queued ones.
     *
     * <p>The engine names this field itself: when its cached figure disagrees
     * with a recount it logs {@code
     * unitCountExcludingBuildingsIncludingQueued:} and prints both. That log
     * line is the whole derivation — nothing here is inferred from the name.
     */
    static final String TEAM_UNIT_COUNT = "w";

    /**
     * The unit cap this game was configured with.
     *
     * <p>Copied onto every player from the game-wide setting at start.
     * {@link #TEAM_UNIT_COUNT} is compared against it on the production path,
     * and a producer at the cap is refused silently (wiki:
     * mechanics-build-actions).
     */
    static final String TEAM_UNIT_CAP = "x";

    /** Current and maximum hit points on an entity. */
    static final String HP = "cu";

    static final String MAX_HP = "cv";

    /**
     * The engine's own per-player visibility test, {@code am.d(n)}.
     *
     * <p>Its body fog-tests the entity's cell against the asking player's fog
     * grid and returns false when the cell reads as hidden. Own units short out
     * before the test. Using it is what keeps perception legitimate: the master
     * entity list holds every unit on the map, so enumerating that directly
     * would give the bot perfect information and stop it playing the same game a
     * human plays (wiki: multiplayer-portability-invariants).
     */
    static final String VISIBLE_TO = "d";

    /** The engine's CommandController instance. */
    static final String CONTROLLER = "cf";

    /**
     * Boolean on the command object that turns a move into an attack-move.
     *
     * <p>Read from the engine's own double-right-click dispatch: {@code
     * gameFramework/f/g.d(float,float,Point)} creates a command, sets {@code
     * e3.h = true}, sets the point and enqueues -- gated on the
     * {@code doubleClickToAttackMove} setting, which is what names the flag.
     */
    static final String ATTACK_MOVE_FLAG = "h";

    /** Credits held by a player. Sits beside the engine's own note to modders. */
    static final String CREDITS = "o";

    /** Engine-assigned object identity, set once at construction. */
    static final String ENTITY_ID = "eh";

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
     * Entity predicate: whether construction has finished.
     *
     * <p>A building appears in the entity list the moment construction starts,
     * so presence is not completion. The distinction is load-bearing twice
     * over: an unfinished factory never advances its production queue, and a
     * planner counting roster entries would call an unfinished structure built
     * (wiki: mechanics-build-actions).
     */
    static final String ENTITY_COMPLETE = "bT";

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
     * Action accessor returning its selector index.
     *
     * <p>The same integer a build command carries. Passing -1 means "whichever
     * action produces this type"; the engine's own lookup compares against this
     * (wiki: building-structures).
     */
    static final String ACTION_INDEX = "t";

    /** Action predicate: whether the given unit may use it right now. */
    static final String ACTION_AVAILABLE = "b";

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
     * The shipped AI, as a player subclass.
     *
     * <p>Only AI players are instances of it. The local human player is
     * constructed as a different subclass entirely, which is why nothing built
     * on this class can survive contact with a human opponent — see
     * {@link AiZones} for why that makes it an instrument rather than a source.
     */
    static final String AI_CLASS = "com.corrodinggames.rts.game.a.a";

    /**
     * The AI's zone list.
     *
     * <p>A concurrent queue, which is what makes reading it from a probe
     * thread the engine's own supported use rather than a race. The AI keeps a
     * second plain-list copy of the same zones; this is the one safe to iterate
     * from off the game thread.
     */
    static final String AI_ZONES = "bm";

    /**
     * Base class of every zone.
     *
     * <p>Pinned so the dump can reject anything else the list might hold. The
     * element type is erased, and reading an unrelated object's fields as a
     * zone would produce a plausible table of numbers (wiki: engine-ai-zones).
     */
    static final String ZONE_CLASS = "com.corrodinggames.rts.game.a.o";

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

    /**
     * Entity predicate: whether it is airborne right now.
     *
     * <p>State, not type. A gunship that has landed answers false and becomes
     * shootable by units that cannot hit aircraft, so this is read per entity
     * per sample rather than derived from the type once.
     */
    static final String ENTITY_FLYING = "i";

    /**
     * Entity predicate: whether it is below the surface right now.
     *
     * <p>State, like {@link #ENTITY_FLYING}: the engine implements it as a
     * height comparison, so a submarine answers true only while submerged and a
     * surfaced one is an ordinary target.
     */
    static final String ENTITY_SUBMERGED = "Q";

    /**
     * Entity predicate: whether it is standing in water.
     *
     * <p>The other half of {@link #UNIT_HITS_LAND_OUT_OF_WATER}. Also a height
     * comparison, so an amphibious unit answers differently on either side of
     * the shoreline.
     */
    static final String ENTITY_TOUCHING_WATER = "cH";

    /**
     * The engine's terrain-connectivity utility.
     *
     * <p>Holds the answer to "can this thing get there", which the bot
     * otherwise has no way to compute: it precomputes connected components per
     * movement layer and reduces reachability to comparing two component ids
     * (wiki: mechanics-movement-layers).
     */
    static final String PATHING_CLASS = "com.corrodinggames.rts.gameFramework.utility.y";

    /**
     * Connectivity lookup: the component id of a world point on one layer.
     *
     * <p>Negative values are not ids. The engine's own reachability predicate
     * treats -1 as impassable and -2 as off-map, and logs -3 as "no
     * isolatedGroups found". Everything here rejects every negative, which is
     * strictly more conservative than the engine — its predicate compares two
     * -3s for equality and answers true.
     */
    static final String PATH_GROUP_AT = "b";

    /**
     * The movement-layer enum: NONE, LAND, BUILDING, AIR, WATER, HOVER,
     * OVER_CLIFF, OVER_CLIFF_WATER.
     *
     * <p>Its constants are resolved by name rather than by field, because the
     * obfuscator renamed the fields and left the names intact
     * (wiki: engine-name-oracle). Matching on {@code "LAND"} therefore survives
     * a reordering that matching on a field letter would not.
     */
    static final String MOVEMENT_CLASS = "com.corrodinggames.rts.game.units.ao";

    /** Layer a land unit travels on, by its engine name. */
    static final String MOVEMENT_LAND = "LAND";

    /**
     * Entity accessor returning its movement layer.
     *
     * <p>Abstract on the entity base, so every unit answers it, and it is what
     * the engine's own reachability check reads to decide which component grid
     * applies.
     */
    static final String ENTITY_MOVEMENT = "h";

    /** The engine's map instance. */
    static final String MAP = "bL";

    /** Map size in tiles. Positions are world units; these are not. */
    static final String MAP_TILES_X = "C";

    static final String MAP_TILES_Y = "D";

    /** Tile size in world units. 20 on desktop, and read rather than assumed. */
    static final String TILE_WIDTH = "n";

    static final String TILE_HEIGHT = "o";

    /** Offset from a tile's origin to its centre, which is where a building goes. */
    static final String TILE_CENTRE_X = "p";

    static final String TILE_CENTRE_Y = "q";

    /**
     * Map accessor returning the item-layer tile at a tile coordinate.
     *
     * <p>The item layer, not the ground layer: resource pools are items. This
     * is the same accessor the engine's own extractor-placement check calls,
     * which is why it is this one and not the ground-layer sibling.
     */
    static final String TILE_AT = "e";

    /**
     * The engine's per-tile fog test, {@code b.a(int, int, n)}.
     *
     * <p>The tile counterpart of {@link #VISIBLE_TO}, and the same predicate:
     * the entity test converts a position to a tile and applies exactly this
     * comparison against the asking player's fog grid. Using it is what stops
     * the bot reading pools through fog it has not lifted.
     */
    static final String TILE_VISIBLE_TO = "a";

    /** Tile flag set by the tileset property {@code res_pool}. */
    static final String TILE_IS_POOL = "i";

    /**
     * Whether the map has fog at all.
     *
     * <p>Read only to report it. Both visibility tests already consult it, so
     * the agent never needs to branch on it -- but whether a run was fogged
     * decides whether "the bot only saw what it could legitimately see" is a
     * property that was tested or merely not violated.
     */
    static final String FOG_ENABLED = "E";

    /** Per-player fog grid. Null when the player has none. */
    static final String FOG_GRID = "N";

    /** Static list of every asset-defined unit type. */
    static final String CUSTOM_TYPE_LIST = "d";

    /**
     * Holds the engine's general-purpose generator and its maths helpers.
     *
     * <p>Not the lockstep path. Anything that must agree between peers goes
     * through a deterministic hash of the match seed and frame counter in the
     * same class; this field is the plain generator everything else uses.
     */
    static final String RANDOM_HOLDER_CLASS = "com.corrodinggames.rts.gameFramework.f";

    /**
     * The unseeded {@code java.util.Random} the opponents' choices draw from.
     *
     * <p>Constructed with no seed, so it takes one from the system clock and
     * every run differs. Seeding it is what makes a measurement repeatable
     * (wiki: policy-combat).
     */
    static final String RANDOM_FIELD = "a";

    /**
     * Player flag: defeated.
     *
     * <p>Set when a player still has units but none that can build or hold
     * ground. The engine names it itself -- the notification it fires on this
     * transition prints "<player> was defeated" -- which is the whole
     * derivation; nothing here is inferred from the letter.
     */
    static final String PLAYER_DEFEATED = "F";

    /**
     * Player flag: wiped out.
     *
     * <p>Set when a player has no units left and no ally holding any. Its
     * notification prints "<player> has been wiped out", and separately "had
     * no starting units" when it happens in the first ten frames.
     */
    static final String PLAYER_WIPED = "G";

    /**
     * Static count of players still in the match.
     *
     * <p>Counts players that are neither absent, nor {@link #PLAYER_DEFEATED},
     * nor {@link #PLAYER_WIPED}. The engine prints it as "N players remaining"
     * and calls its own end-of-match hook when it reaches one, so this is the
     * engine's verdict rather than our arithmetic (wiki: policy-grading).
     */
    static final String PLAYERS_REMAINING = "g";

    /**
     * The playable-slot array the engine's own survivor count walks.
     *
     * <p>There are two player arrays and only this one is the roster. The other
     * is a display list that also carries the neutral and observer pseudo-teams,
     * so counting it would report players the match does not have. This is the
     * array {@link #PLAYERS_REMAINING} itself iterates, which is what pins it.
     */
    static final String TEAM_ROSTER = "as";

    /** How many slots of {@link #TEAM_ROSTER} are in play. */
    static final String TEAM_ROSTER_SIZE = "c";

    /** Team predicate: whether the slot is empty rather than a player. */
    static final String TEAM_ABSENT = "b";

    /**
     * The engine's own per-player statistic, as an enum of readable names.
     *
     * <p>The bot measured income by regressing a credit balance against the
     * clock across windows in which it deliberately bought nothing. The engine
     * has the figure. This enum is how it asks for it, and its constants name
     * themselves — {@code income}, {@code armyValue}, {@code buildingValue},
     * {@code credits} — because ProGuard renamed the fields and left the
     * constant name strings alone (wiki: engine-name-oracle).
     *
     * <p>Matching on those names rather than on ordinal is the point: a
     * reordered enum would silently swap army value for income, and a renamed
     * one fails loudly instead.
     */
    static final String PLAYER_STAT_CLASS = "com.corrodinggames.rts.gameFramework.g.f";

    /** Reads one statistic for one player. Takes a {@link #TEAM_CLASS}. */
    static final String PLAYER_STAT_READ = "a";

    /** {@link #PLAYER_STAT_CLASS} constant: credits earned per second. */
    static final String STAT_INCOME = "income";

    /** {@link #PLAYER_STAT_CLASS} constant: the value of everything mobile. */
    static final String STAT_ARMY_VALUE = "armyValue";

    /** {@link #PLAYER_STAT_CLASS} constant: the value of everything standing. */
    static final String STAT_BUILDING_VALUE = "buildingValue";

    static final String PIN = " -- pinned build is 1.15 (code 176, build #28)";
}
