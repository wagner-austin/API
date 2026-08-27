"""The encoded fingerprint a stored cloze record carries.

Lifted out of ``test_cloze_api`` when the fingerprint grew a host and a
package axis: the file was already at the 600-line ceiling, and a fixture
that every axis addition lengthens does not belong inside the suite it
serves.

Built through :func:`platform_core.testing.sample_run_fingerprint` rather than
written as a JSON literal, which is the point. The literal it replaced listed
four axes; when the fingerprint grew to six it silently described an
incomplete configuration and every test using it failed at the decoder. A
fixture that goes through the canonical builder cannot fall behind the type.
"""

from __future__ import annotations

from platform_core.comparability import encode_run_fingerprint
from platform_core.determinism_record import determinism_record
from platform_core.json_utils import JSONValue
from platform_core.testing import sample_run_fingerprint

#: What a completed cloze record must carry beside the number: the
#: configuration it was produced under. An accuracy without one cannot be
#: compared with any other measurement, because a disagreement is
#: indistinguishable from a working image scored on a different card.
CLOZE_FINGERPRINT: JSONValue = encode_run_fingerprint(
    sample_run_fingerprint(
        image_digest="sha256:abc",
        gpu_model="NVIDIA GeForce RTX 3090 Ti",
        driver_version="591.86",
        determinism=determinism_record(
            "torch",
            {
                "deterministic_algorithms": "true",
                "cublas_workspace_config": ":4096:8",
                "matmul_tf32": "false",
                "cudnn_tf32": "false",
                "cudnn_deterministic": "true",
                "cudnn_benchmark": "false",
            },
        ),
    )
)

__all__ = ["CLOZE_FINGERPRINT"]
