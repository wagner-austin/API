"""Real-time pygame preview for procart neon orbs with keyboard controls.

Controls:
  1/2: Orb count -/+
  3/4: Core radius -/+
  5/6: Halo radius -/+
  7/8: Core intensity -/+
  9/0: Halo intensity -/+
  -/=: Speed -/+
  R: Reset defaults
  ESC: Quit
"""
from __future__ import annotations

import numpy as np
import pygame

from procart.color import apply_tone_map
from procart.modules.background import BlackBackground
from procart.modules.neon_orbs import NeonOrbs
from procart.scene import Layer
from procart.types import (
    NeonGlowConfig,
    NeonOrbsLayerConfig,
    Resolution,
    ToneMappingConfigExposureGamma,
)


def render_frame(
    t: float,
    resolution: Resolution,
    orb_count: int,
    core_radius: float,
    halo_radius: float,
    core_intensity: float,
    halo_intensity: float,
) -> np.ndarray:
    """Render a single frame with current parameters."""
    # Black background
    bg_mod = BlackBackground()

    # Neon orbs with current params
    glow: NeonGlowConfig = {
        "core_intensity": core_intensity,
        "halo_intensity": halo_intensity,
        "core_radius": core_radius,
        "halo_radius": halo_radius,
    }
    orbs_cfg: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "orbs",
        "opacity": 1.0,
        "parallax_depth": 1.0,
        "composite": "normal",
        "count": orb_count,
        "glow": glow,
    }
    orbs_mod = NeonOrbs(orbs_cfg)

    # Render layers
    bg_rgba = bg_mod.render_frame(t, resolution, 0, 0.0, 0.0)
    orbs_rgba = orbs_mod.render_frame(t, resolution, 0, 0.0, 0.0)

    # Alpha composite (orbs over background)
    bg_r, bg_g, bg_b = bg_rgba[:, :, 0], bg_rgba[:, :, 1], bg_rgba[:, :, 2]
    o_r, o_g, o_b, o_a = orbs_rgba[:, :, 0], orbs_rgba[:, :, 1], orbs_rgba[:, :, 2], orbs_rgba[:, :, 3]

    inv_a = 1.0 - o_a
    r = o_r + bg_r * inv_a
    g = o_g + bg_g * inv_a
    b = o_b + bg_b * inv_a

    rgb_hdr = np.stack([r, g, b], axis=-1)

    # Tone map
    tone_cfg: ToneMappingConfigExposureGamma = {
        "type": "exposure_gamma",
        "exposure": 1.0,
        "gamma": 2.2,
    }
    rgb_ldr = apply_tone_map(rgb_hdr, tone_cfg)

    # Convert to uint8 for pygame
    rgb_uint8 = (np.clip(rgb_ldr, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb_uint8


def main() -> int:
    pygame.init()

    render_w, render_h = 512, 512
    screen = pygame.display.set_mode((render_w, render_h))
    pygame.display.set_caption("Procart Live - ESC quit, see console for controls")

    font = pygame.font.Font(None, 24)

    # Default parameters
    defaults = {
        "orb_count": 3,
        "core_radius": 0.03,
        "halo_radius": 0.15,
        "core_intensity": 2.0,
        "halo_intensity": 0.4,
        "speed": 0.5,
    }
    params = defaults.copy()

    clock = pygame.time.Clock()
    t = 0.0
    running = True

    resolution: Resolution = {"width": render_w, "height": render_h}

    print(__doc__)

    while running:
        time_delta = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Orb count
                elif event.key == pygame.K_1:
                    params["orb_count"] = max(1, params["orb_count"] - 1)
                elif event.key == pygame.K_2:
                    params["orb_count"] = min(20, params["orb_count"] + 1)
                # Core radius
                elif event.key == pygame.K_3:
                    params["core_radius"] = max(0.01, params["core_radius"] - 0.005)
                elif event.key == pygame.K_4:
                    params["core_radius"] = min(0.2, params["core_radius"] + 0.005)
                # Halo radius
                elif event.key == pygame.K_5:
                    params["halo_radius"] = max(0.02, params["halo_radius"] - 0.01)
                elif event.key == pygame.K_6:
                    params["halo_radius"] = min(0.5, params["halo_radius"] + 0.01)
                # Core intensity
                elif event.key == pygame.K_7:
                    params["core_intensity"] = max(0.1, params["core_intensity"] - 0.2)
                elif event.key == pygame.K_8:
                    params["core_intensity"] = min(10.0, params["core_intensity"] + 0.2)
                # Halo intensity
                elif event.key == pygame.K_9:
                    params["halo_intensity"] = max(0.05, params["halo_intensity"] - 0.1)
                elif event.key == pygame.K_0:
                    params["halo_intensity"] = min(3.0, params["halo_intensity"] + 0.1)
                # Speed
                elif event.key == pygame.K_MINUS:
                    params["speed"] = max(0.1, params["speed"] - 0.1)
                elif event.key == pygame.K_EQUALS:
                    params["speed"] = min(3.0, params["speed"] + 0.1)
                # Reset
                elif event.key == pygame.K_r:
                    params = defaults.copy()

        # Update time (looping)
        t += time_delta * params["speed"]
        t = t % 1.0

        # Render frame
        rgb = render_frame(
            t,
            resolution,
            int(params["orb_count"]),
            params["core_radius"],
            params["halo_radius"],
            params["core_intensity"],
            params["halo_intensity"],
        )

        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        # Draw overlay text
        fps = clock.get_fps()
        lines = [
            f"FPS: {fps:.1f}",
            f"Orbs(1/2): {params['orb_count']}",
            f"Core R(3/4): {params['core_radius']:.3f}",
            f"Halo R(5/6): {params['halo_radius']:.2f}",
            f"Core I(7/8): {params['core_intensity']:.1f}",
            f"Halo I(9/0): {params['halo_intensity']:.2f}",
            f"Speed(-/=): {params['speed']:.1f}",
            "R=reset ESC=quit",
        ]
        y = 10
        for line in lines:
            text = font.render(line, True, (255, 255, 255), (0, 0, 0))
            screen.blit(text, (10, y))
            y += 20

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
