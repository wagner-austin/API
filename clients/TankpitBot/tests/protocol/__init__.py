"""Protocol test package.

Tests are organized by message type and functionality:
- test_enums.py: Enum tests (Rank, Team, Equipment, TerrainType)
- test_helpers.py: Helper function tests (x16, x24, require_*)
- test_text.py: Text message decoder tests
- test_combat.py: Combat decoder tests (shoot, hit, deactivation, mines)
- test_resources.py: Resource decoder tests (fuel, inventory, equipment)
- test_radar.py: Radar decoder tests and validators
- test_movement.py: Movement decoder tests
- test_tank.py: Tank decoder tests (info, entry, exit, status)
- test_world.py: World decoder tests (viewport, terrain, sync, container)
- test_dispatcher.py: Main dispatcher and message type detection tests
"""
