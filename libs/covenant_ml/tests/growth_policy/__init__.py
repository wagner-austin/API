"""Tests for the growth-policy experiment package.

One module per source module. The layers that reach a number are driven against
the real learners and the real scikit-learn metrics rather than stand-ins: this
package exists to produce figures that go into a document, and a figure taken
from a stand-in would report the stand-in.
"""

from __future__ import annotations
