package rwbot.agent;

import java.lang.reflect.Method;

/**
 * The map as the planner needs to see it: where the resource pools are.
 *
 * <p>Resource pools are terrain, not units. They never appear in the entity
 * list, so a planner reading only the roster cannot see them at all — and an
 * extractor is the one structure the engine refuses to place anywhere else
 * (wiki: mechanics-resource-pools).
 *
 * <p>A pool is a tile whose tileset declared the property {@code res_pool}. The
 * loader turns that property into a boolean on the tile, the extractor
 * placement check reads that boolean, and this class reads the same one. No
 * step of that chain is inferred.
 *
 * <p><b>Scanned once, filtered every sample.</b> The pool tiles of a map are
 * fixed for its lifetime, and scanning a 230x230 grid reflectively is not
 * something to do four times a second. What does change is which of them the
 * player can see, so the scan is cached against the map instance and only the
 * fog test runs per sample.
 */
final class MapTiles {

    /**
     * A resource pool: where it is on the tile grid, and the world point at its
     * centre.
     *
     * <p>Both are carried because they answer different questions. The tile
     * coordinate identifies the pool — it is stable, integral, and the unit in
     * which the engine's own placement check works. The world point is where a
     * build order has to be addressed, because orders are in world space.
     */
    static final class Pool {

        private final int tileX;
        private final int tileY;
        private final float x;
        private final float y;

        Pool(int tileX, int tileY, float x, float y) {
            this.tileX = tileX;
            this.tileY = tileY;
            this.x = x;
            this.y = y;
        }

        int tileX() {
            return tileX;
        }

        int tileY() {
            return tileY;
        }

        float x() {
            return x;
        }

        float y() {
            return y;
        }
    }

    /**
     * The map instance the cache was built from, or null before any scan.
     *
     * <p>Identity rather than a flag: a new match means a new map object, so
     * comparing the reference is what makes the cache correct across a map
     * change without anything having to remember to invalidate it.
     *
     * <p>Not synchronised, and deliberately so. Every caller reaches this
     * through the game thread, which is also the only thread allowed to read
     * the simulation. A lock here would suggest a second caller exists.
     */
    private static Object scannedMap;

    private static java.util.List<Pool> scannedPools = java.util.Collections.emptyList();

    private MapTiles() {
    }

    /**
     * Lists every resource pool the current player can legitimately see.
     *
     * <p>Filtered by the engine's own per-tile fog test, which is the same
     * predicate its entity visibility test applies after converting a position
     * to a tile. Reading the unfiltered grid would hand the planner every pool
     * on the map from the first frame, which is not a view a human has (wiki:
     * multiplayer-portability-invariants).
     *
     * <p>Must run on the game thread.
     *
     * @param engine The live engine instance.
     * @return The visible pools, in scan order. Empty when no map is loaded or
     *     there is no current player.
     */
    static java.util.List<Pool> visiblePools(Object engine) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        Object map = EngineAccess.readField(engine, EngineNames.MAP);
        if (team == null || map == null) {
            return java.util.Collections.emptyList();
        }
        java.util.List<Pool> all = pools(engine, map);
        Method visibleTo =
                EngineAccess.pinnedMethod(
                        map.getClass(),
                        EngineNames.TILE_VISIBLE_TO,
                        int.class,
                        int.class,
                        EngineAccess.pinnedClass(EngineNames.TEAM_CLASS));
        java.util.List<Pool> visible = new java.util.ArrayList<Pool>();
        for (Pool pool : all) {
            Object seen =
                    EngineAccess.invoke(
                            visibleTo, map, Integer.valueOf(pool.tileX()),
                            Integer.valueOf(pool.tileY()), team);
            if (Boolean.TRUE.equals(seen)) {
                visible.add(pool);
            }
        }
        return visible;
    }

    /**
     * Returns the map's pool tiles, scanning it the first time it is seen.
     *
     * @param engine The live engine instance.
     * @param map The engine's map instance.
     * @return Every pool on the map, visible or not.
     */
    private static java.util.List<Pool> pools(Object engine, Object map) {
        if (map != scannedMap) {
            scannedPools = scan(map);
            scannedMap = map;
            Log.info("map scan: " + scannedPools.size() + " resource pool(s)");
            Log.info(describeFog(engine));
        }
        return scannedPools;
    }

    /**
     * Reports whether the fog filter is actually in force for a player.
     *
     * <p>Both visibility tests short out when the map has no fog or the player
     * has no fog grid, and they do so silently -- which is correct behaviour
     * and a reporting problem. Without this, a run in which everything was
     * visible is indistinguishable from a run in which the filter was working
     * and happened to reveal everything, and only one of those tells you the
     * bot is playing fairly (wiki: perception-visibility).
     *
     * @param engine The live engine instance.
     * @return A short description of the fog state.
     */
    static String describeFog(Object engine) {
        Object team = EngineAccess.readField(engine, EngineNames.LOCAL_TEAM);
        Object map = EngineAccess.readField(engine, EngineNames.MAP);
        if (team == null || map == null) {
            return "fog: no player or no map";
        }
        boolean enabled = EngineAccess.readBooleanField(map, EngineNames.FOG_ENABLED);
        Object grid = EngineAccess.readField(team, EngineNames.FOG_GRID);
        if (!enabled) {
            return "fog: DISABLED on this map -- every visibility test passes";
        }
        if (grid == null) {
            return "fog: enabled, but this player has no fog grid -- every test passes";
        }
        return "fog: enabled and this player has a grid -- visibility is filtered";
    }

    /**
     * Walks the whole tile grid once, collecting the pools.
     *
     * <p>The item layer is consulted rather than the ground layer, because that
     * is the layer the engine's own extractor check reads. A tile coordinate
     * with nothing on that layer reads as null, which is the ordinary answer
     * for most of a map.
     *
     * @param map The engine's map instance.
     * @return Every pool on the map, in row-major scan order.
     */
    private static java.util.List<Pool> scan(Object map) {
        int tilesX = EngineAccess.readIntField(map, EngineNames.MAP_TILES_X);
        int tilesY = EngineAccess.readIntField(map, EngineNames.MAP_TILES_Y);
        int tileWidth = EngineAccess.readIntField(map, EngineNames.TILE_WIDTH);
        int tileHeight = EngineAccess.readIntField(map, EngineNames.TILE_HEIGHT);
        int centreX = EngineAccess.readIntField(map, EngineNames.TILE_CENTRE_X);
        int centreY = EngineAccess.readIntField(map, EngineNames.TILE_CENTRE_Y);
        Method tileAt =
                EngineAccess.pinnedMethod(
                        map.getClass(), EngineNames.TILE_AT, int.class, int.class);

        java.util.List<Pool> found = new java.util.ArrayList<Pool>();
        for (int tileY = 0; tileY < tilesY; tileY++) {
            for (int tileX = 0; tileX < tilesX; tileX++) {
                Object tile =
                        EngineAccess.invoke(
                                tileAt, map, Integer.valueOf(tileX), Integer.valueOf(tileY));
                if (tile == null
                        || !EngineAccess.readBooleanField(tile, EngineNames.TILE_IS_POOL)) {
                    continue;
                }
                found.add(
                        new Pool(
                                tileX,
                                tileY,
                                (float) (tileX * tileWidth + centreX),
                                (float) (tileY * tileHeight + centreY)));
            }
        }
        return found;
    }
}
