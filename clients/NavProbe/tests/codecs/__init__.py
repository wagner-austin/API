"""Tests for the per-record codecs.

One module per codec module, mirroring :mod:`navprobe.codecs`. Each asserts the
encoded field order, that the round trip is the identity, and that every
malformed shape the decoder documents as refused is in fact refused by code.
"""

from __future__ import annotations
