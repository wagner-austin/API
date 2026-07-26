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

    static final String PIN = " -- pinned build is 1.15 (code 176, build #28)";
}
