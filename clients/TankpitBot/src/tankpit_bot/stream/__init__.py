"""Display-capture video: Xvfb, ffmpeg, and the HLS files they produce.

The public demo shows anonymous strangers a bot playing. The pipeline
that does it captures the bot's OWN rendered display and encodes real
video, replacing the in-page canvas scraper that spent 2026-09-04
proving it is the wrong class of system (board post 21:25Z, session
23fe4130): scraping re-encoded frames on the game's main thread capped
the capture rate, and MJPEG over one endless HTTP response had no way
to say "nothing changed" and no congestion story at all.

Five modules, one concern each:

* :mod:`tankpit_bot.stream.types` — the capture configuration shape.
* :mod:`tankpit_bot.stream.helper_process` — the socket readiness wait
  and the terminate-then-kill stop every helper process shares.
* :mod:`tankpit_bot.stream.audio` — the per-bot PulseAudio server the
  game's sound plays into.
* :mod:`tankpit_bot.stream.capture` — Xvfb + ffmpeg process lifecycle;
  ffmpeg records the display and the sound into one stream.
* :mod:`tankpit_bot.stream.hls` — reading the produced HLS files for
  an HTTP response, shared by the child's own surface and the fleet
  manager's demo routes (same container, same filesystem — the
  manager serves segments straight from disk, no relay).
"""
