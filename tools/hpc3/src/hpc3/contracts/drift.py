"""The drift contract: how far a run's code is from the code its submitter has.

An image is built from one commit and carries it at ``GIT_COMMIT``; the
submitter's checkout moves on. A run that starts a module the image HAS is
admitted, because reproducing an old command under its old image must stay
legal, but the code it executes is the image's, not HEAD's. When that
module's source has changed in between, the numbers the run produces come
from older code than the person reading them is likely to assume, and
nothing else in the record says so (board task 7aab32ef, following
2c8161a3).

So a submission records, beside the image's digest, which commit built the
image, which HEAD it was compared with, and per module what was compared and
what changed. The record says how much it examined: a module that was not
compared carries why, so an empty ``changed`` reads as "compared and
unchanged" only where something was compared.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_list,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict


class ModuleDrift(TypedDict):
    """How one module a run starts compares with the submitter's HEAD.

    Attributes:
        directory: The directory the command starts it from, which with
            ``module`` names one invocation.
        module: The dotted module name.
        compared: Repository-relative paths holding its top-level package's
            source at HEAD, which is what was compared. Empty when it was
            not compared.
        changed: Repository-relative paths under ``compared`` that differ
            between the image's commit and HEAD. Empty when nothing changed
            or nothing was compared.
        not_compared: Why it was not compared, or the empty string when it
            was.
    """

    directory: str
    module: str
    compared: list[str]
    changed: list[str]
    not_compared: str


class ImageDrift(TypedDict):
    """A submission's code compared with the code its submitter has.

    Attributes:
        image_commit: The commit the run's image was built from, or the
            empty string when the run has no image, which is a positive fact:
            a directory environment is identified by no build.
        head: The submitter's repository HEAD it was compared with, or the
            empty string when the run has no image.
        modules: One entry per ``python -m`` invocation the run starts.
    """

    image_commit: str
    head: str
    modules: list[ModuleDrift]


def _encode_module(module: ModuleDrift) -> dict[str, JSONValue]:
    """Encode one module's entry.

    Args:
        module: Entry to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    compared: list[JSONValue] = list(module["compared"])
    changed: list[JSONValue] = list(module["changed"])
    return {
        "directory": module["directory"],
        "module": module["module"],
        "compared": compared,
        "changed": changed,
        "not_compared": module["not_compared"],
    }


def encode_image_drift(drift: ImageDrift) -> dict[str, JSONValue]:
    """Encode a drift record to a JSON object.

    Args:
        drift: Record to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    modules: list[JSONValue] = [_encode_module(module) for module in drift["modules"]]
    return {"image_commit": drift["image_commit"], "head": drift["head"], "modules": modules}


def _decode_module(value: JSONValue) -> ModuleDrift:
    """Decode one module's entry.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated entry.

    Raises:
        JSONTypeError: If a field is missing or mistyped, or the entry both
            names a reason it was not compared and lists what was compared.
    """
    obj = narrow_json_to_dict(value)
    module = ModuleDrift(
        directory=require_str(obj, "directory"),
        module=require_str(obj, "module"),
        compared=require_str_list(obj, "compared"),
        changed=require_str_list(obj, "changed"),
        not_compared=require_str(obj, "not_compared"),
    )
    if (module["not_compared"] == "") == (module["compared"] == []):
        raise JSONTypeError(
            f"module {module['module']!r} must either list what was compared or say why "
            "it was not, and not both"
        )
    return module


def decode_image_drift(value: JSONValue) -> ImageDrift:
    """Decode and validate a drift record.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or a run with no image names a HEAD it was compared
            with.
    """
    obj = narrow_json_to_dict(value)
    drift = ImageDrift(
        image_commit=require_str(obj, "image_commit"),
        head=require_str(obj, "head"),
        modules=[_decode_module(module) for module in require_list(obj, "modules")],
    )
    if (drift["image_commit"] == "") != (drift["head"] == ""):
        raise JSONTypeError(
            "'image_commit' and 'head' are both empty for a run with no image, and both "
            f"set for one with an image; got {drift['image_commit']!r} and {drift['head']!r}"
        )
    return drift


__all__ = ["ImageDrift", "ModuleDrift", "decode_image_drift", "encode_image_drift"]
