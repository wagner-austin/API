"""Tests for cleargbm.types: buffer payloads."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    FloatBufferData,
    HistogramBufferData,
    IntBufferData,
    JSONDict,
    JSONTypeError,
    decode_float_buffer_data,
    decode_histogram_buffer_data,
    decode_int_buffer_data,
    encode_float_buffer_data,
    encode_histogram_buffer_data,
    encode_int_buffer_data,
)

# =============================================================================
# Buffer Type Tests
# =============================================================================


class TestFloatBufferData:
    """Tests for FloatBufferData encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        data: FloatBufferData = FloatBufferData(
            values=(1.0, 2.0, 3.0),
            size=3,
        )
        encoded: JSONDict = encode_float_buffer_data(data["values"], data["size"])
        decoded: FloatBufferData = decode_float_buffer_data(encoded)
        assert decoded["values"] == (1.0, 2.0, 3.0)
        assert decoded["size"] == 3

    def test_decode_coerces_int_to_float(self) -> None:
        """Test decode coerces int values to float."""
        raw: JSONDict = {"values": [1, 2, 3], "size": 3}
        decoded: FloatBufferData = decode_float_buffer_data(raw)
        assert decoded["values"] == (1.0, 2.0, 3.0)

    def test_decode_raises_on_missing_size(self) -> None:
        """Test decode raises KeyError for missing size."""
        raw: JSONDict = {"values": [1.0]}
        with pytest.raises(KeyError):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_missing_values(self) -> None:
        """Test decode raises KeyError for missing values."""
        raw: JSONDict = {"size": 3}
        with pytest.raises(KeyError):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_non_list_values(self) -> None:
        """Test decode raises JSONTypeError for non-list values."""
        raw: JSONDict = {"values": "not a list", "size": 3}
        with pytest.raises(JSONTypeError, match="values must be list"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_bool_value(self) -> None:
        """Test decode raises JSONTypeError for bool in values."""
        raw: JSONDict = {"values": [True, 2.0], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[0\] must be float, got bool"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_string_value(self) -> None:
        """Test decode raises JSONTypeError for string in values."""
        raw: JSONDict = {"values": [1.0, "not a float"], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[1\] must be float"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_size_mismatch(self) -> None:
        """Test decode raises ValueError for values/size mismatch."""
        raw: JSONDict = {"values": [1.0, 2.0], "size": 3}
        with pytest.raises(ValueError, match="values length 2 != size 3"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_non_positive_size(self) -> None:
        """Test decode raises ValueError for non-positive size."""
        raw: JSONDict = {"values": [], "size": 0}
        with pytest.raises(ValueError, match="size must be positive"):
            decode_float_buffer_data(raw)


class TestIntBufferData:
    """Tests for IntBufferData encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        data: IntBufferData = IntBufferData(
            values=(1, 2, 3),
            size=3,
        )
        encoded: JSONDict = encode_int_buffer_data(data["values"], data["size"])
        decoded: IntBufferData = decode_int_buffer_data(encoded)
        assert decoded["values"] == (1, 2, 3)
        assert decoded["size"] == 3

    def test_decode_raises_on_missing_size(self) -> None:
        """Test decode raises KeyError for missing size."""
        raw: JSONDict = {"values": [1]}
        with pytest.raises(KeyError):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_missing_values(self) -> None:
        """Test decode raises KeyError for missing values."""
        raw: JSONDict = {"size": 3}
        with pytest.raises(KeyError):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_non_list_values(self) -> None:
        """Test decode raises JSONTypeError for non-list values."""
        raw: JSONDict = {"values": "not a list", "size": 3}
        with pytest.raises(JSONTypeError, match="values must be list"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_bool_value(self) -> None:
        """Test decode raises JSONTypeError for bool in values."""
        raw: JSONDict = {"values": [True, 2], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[0\] must be int"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_float_value(self) -> None:
        """Test decode raises JSONTypeError for float in values."""
        raw: JSONDict = {"values": [1, 2.5], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[1\] must be int"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_size_mismatch(self) -> None:
        """Test decode raises ValueError for values/size mismatch."""
        raw: JSONDict = {"values": [1, 2], "size": 3}
        with pytest.raises(ValueError, match="values length 2 != size 3"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_non_positive_size(self) -> None:
        """Test decode raises ValueError for non-positive size."""
        raw: JSONDict = {"values": [], "size": 0}
        with pytest.raises(ValueError, match="size must be positive"):
            decode_int_buffer_data(raw)


class TestHistogramBufferData:
    """Tests for HistogramBufferData encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        data: HistogramBufferData = HistogramBufferData(
            gradient_sums=(1.0, 2.0, 3.0),
            hessian_sums=(0.5, 1.0, 1.5),
            counts=(1, 2, 3),
            n_bins=3,
        )
        encoded: JSONDict = encode_histogram_buffer_data(
            data["gradient_sums"],
            data["hessian_sums"],
            data["counts"],
            data["n_bins"],
        )
        decoded: HistogramBufferData = decode_histogram_buffer_data(encoded)
        assert decoded["gradient_sums"] == (1.0, 2.0, 3.0)
        assert decoded["hessian_sums"] == (0.5, 1.0, 1.5)
        assert decoded["counts"] == (1, 2, 3)
        assert decoded["n_bins"] == 3

    def test_decode_coerces_int_to_float(self) -> None:
        """Test decode coerces int values to float for gradient/hessian sums."""
        raw: JSONDict = {
            "gradient_sums": [1, 2, 3],
            "hessian_sums": [1, 2, 3],
            "counts": [1, 2, 3],
            "n_bins": 3,
        }
        decoded: HistogramBufferData = decode_histogram_buffer_data(raw)
        assert decoded["gradient_sums"] == (1.0, 2.0, 3.0)
        assert decoded["hessian_sums"] == (1.0, 2.0, 3.0)

    def test_decode_raises_on_missing_gradient_sums(self) -> None:
        """Test decode raises KeyError for missing gradient_sums."""
        raw: JSONDict = {"hessian_sums": [1.0], "counts": [1], "n_bins": 1}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_missing_hessian_sums(self) -> None:
        """Test decode raises KeyError for missing hessian_sums."""
        raw: JSONDict = {"gradient_sums": [1.0], "counts": [1], "n_bins": 1}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_missing_counts(self) -> None:
        """Test decode raises KeyError for missing counts."""
        raw: JSONDict = {"gradient_sums": [1.0], "hessian_sums": [1.0], "n_bins": 1}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_missing_n_bins(self) -> None:
        """Test decode raises KeyError for missing n_bins."""
        raw: JSONDict = {"gradient_sums": [1.0], "hessian_sums": [1.0], "counts": [1]}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_list_gradient_sums(self) -> None:
        """Test decode raises JSONTypeError for non-list gradient_sums."""
        raw: JSONDict = {
            "gradient_sums": "not a list",
            "hessian_sums": [1.0],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match="gradient_sums must be list"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_list_hessian_sums(self) -> None:
        """Test decode raises JSONTypeError for non-list hessian_sums."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": "not a list",
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match="hessian_sums must be list"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_list_counts(self) -> None:
        """Test decode raises JSONTypeError for non-list counts."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [1.0],
            "counts": "not a list",
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match="counts must be list"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_bool_in_gradient_sums(self) -> None:
        """Test decode raises JSONTypeError for bool in gradient_sums."""
        raw: JSONDict = {
            "gradient_sums": [True],
            "hessian_sums": [1.0],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"gradient_sums\[0\] must be float, got bool"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_string_in_gradient_sums(self) -> None:
        """Test decode raises JSONTypeError for string in gradient_sums."""
        raw: JSONDict = {
            "gradient_sums": ["not a float"],
            "hessian_sums": [1.0],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"gradient_sums\[0\] must be float"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_bool_in_hessian_sums(self) -> None:
        """Test decode raises JSONTypeError for bool in hessian_sums."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [True],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"hessian_sums\[0\] must be float, got bool"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_string_in_hessian_sums(self) -> None:
        """Test decode raises JSONTypeError for string in hessian_sums."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": ["not a float"],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"hessian_sums\[0\] must be float"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_bool_in_counts(self) -> None:
        """Test decode raises JSONTypeError for bool in counts."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [1.0],
            "counts": [True],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"counts\[0\] must be int"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_float_in_counts(self) -> None:
        """Test decode raises JSONTypeError for float in counts."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [1.0],
            "counts": [1.5],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"counts\[0\] must be int"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_gradient_sums_length_mismatch(self) -> None:
        """Test decode raises ValueError for gradient_sums/n_bins mismatch."""
        raw: JSONDict = {
            "gradient_sums": [1.0, 2.0],
            "hessian_sums": [1.0, 2.0, 3.0],
            "counts": [1, 2, 3],
            "n_bins": 3,
        }
        with pytest.raises(ValueError, match="gradient_sums length 2 != n_bins 3"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_hessian_sums_length_mismatch(self) -> None:
        """Test decode raises ValueError for hessian_sums/n_bins mismatch."""
        raw: JSONDict = {
            "gradient_sums": [1.0, 2.0, 3.0],
            "hessian_sums": [1.0, 2.0],
            "counts": [1, 2, 3],
            "n_bins": 3,
        }
        with pytest.raises(ValueError, match="hessian_sums length 2 != n_bins 3"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_counts_length_mismatch(self) -> None:
        """Test decode raises ValueError for counts/n_bins mismatch."""
        raw: JSONDict = {
            "gradient_sums": [1.0, 2.0, 3.0],
            "hessian_sums": [1.0, 2.0, 3.0],
            "counts": [1, 2],
            "n_bins": 3,
        }
        with pytest.raises(ValueError, match="counts length 2 != n_bins 3"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_positive_n_bins(self) -> None:
        """Test decode raises ValueError for non-positive n_bins."""
        raw: JSONDict = {
            "gradient_sums": [],
            "hessian_sums": [],
            "counts": [],
            "n_bins": 0,
        }
        with pytest.raises(ValueError, match="n_bins must be positive"):
            decode_histogram_buffer_data(raw)
