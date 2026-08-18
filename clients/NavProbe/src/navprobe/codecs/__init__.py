"""One codec module per record type.

Every module here declares its own banner and its own header field count, and
builds entirely out of :mod:`navprobe.wireformat`. Separate banners are what
stop one record being read as another: the record types have overlapping field
names, so without the banner a trial's first header line would decode as a run's
and the loader would return a record that never existed.

Nothing is re-exported from this package. A caller imports the codec for the
record it is handling, which keeps the dependency visible at the import site
rather than hidden behind a shared namespace.
"""

from __future__ import annotations

__all__: list[str] = []
