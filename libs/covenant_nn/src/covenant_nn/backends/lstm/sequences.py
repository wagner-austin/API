"""Sequence building utilities for LSTM temporal modeling.

Converts flat firm-year data into temporal sequences for LSTM training.
Each sequence contains multiple consecutive years of data for a single entity.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class SequenceData(TypedDict):
    """Result of sequence building."""

    x_sequences: NDArray[np.float64]  # Shape: (n_sequences, seq_len, n_features)
    y_sequences: NDArray[np.int64]  # Shape: (n_sequences,) - label for final year
    sequence_entity_ids: NDArray[np.int64]  # Entity ID for each sequence
    n_sequences: int
    n_features: int
    sequence_length: int


def build_sequences(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    entity_ids: NDArray[np.int64],
    years: NDArray[np.int64],
    sequence_length: int,
) -> SequenceData:
    """Build temporal sequences from flat firm-year data.

    Groups data by entity, orders by year, and creates sliding window sequences.
    Each sequence contains `sequence_length` consecutive years of data for one entity.
    The label for each sequence is the label of the final year in the sequence.

    Args:
        x_features: Feature matrix, shape (n_samples, n_features).
        y_labels: Binary labels, shape (n_samples,).
        entity_ids: Entity identifier for each sample, shape (n_samples,).
        years: Year for each sample, shape (n_samples,).
        sequence_length: Number of years to include in each sequence.

    Returns:
        SequenceData with x_sequences, y_sequences, and metadata.

    Raises:
        ValueError: If sequence_length < 1 or no valid sequences can be built.
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    n_features: int = int(x_features.shape[1])

    # Get unique entities as a list of ints
    # Build list with explicit loop to avoid Any type issues from indexing
    unique_entities_arr: NDArray[np.int64] = np.unique(entity_ids)
    n_unique: int = int(unique_entities_arr.shape[0])
    unique_entity_list: list[int] = []
    for i in range(n_unique):
        # np.asarray().flat gives typed iterator; take first element
        flat_iter = np.asarray(unique_entities_arr[i : i + 1], dtype=np.int64).flat
        val: int = int(flat_iter[0])
        unique_entity_list.append(val)

    # Build sequences for each entity
    all_x_seqs: list[NDArray[np.float64]] = []
    all_y_seqs: list[int] = []
    all_entity_seqs: list[int] = []

    for entity_id_int in unique_entity_list:
        # Get indices for this entity
        entity_mask: NDArray[np.bool_] = entity_ids == entity_id_int
        entity_indices: NDArray[np.int64] = np.where(entity_mask)[0].astype(np.int64)

        if int(entity_indices.shape[0]) < sequence_length:
            # Not enough years for this entity
            continue

        # Sort by year
        entity_years: NDArray[np.int64] = years[entity_indices]
        sort_order: NDArray[np.int64] = np.argsort(entity_years).astype(np.int64)
        sorted_indices: NDArray[np.int64] = entity_indices[sort_order]

        # Create sliding window sequences
        n_entity_samples: int = int(sorted_indices.shape[0])
        for start in range(n_entity_samples - sequence_length + 1):
            end = start + sequence_length
            seq_indices: NDArray[np.int64] = sorted_indices[start:end]

            # Extract features for this sequence
            x_seq: NDArray[np.float64] = x_features[seq_indices]  # (seq_len, n_features)
            # Label is the label of the final year in the sequence
            # Use slice indexing + flat to get typed int (avoids Any from scalar indexing)
            seq_len_idx: int = int(seq_indices.shape[0]) - 1
            last_idx_flat = np.asarray(
                seq_indices[seq_len_idx : seq_len_idx + 1], dtype=np.int64
            ).flat
            last_idx: int = int(last_idx_flat[0])
            y_val_flat = np.asarray(y_labels[last_idx : last_idx + 1], dtype=np.int64).flat
            y_seq: int = int(y_val_flat[0])

            all_x_seqs.append(x_seq)
            all_y_seqs.append(y_seq)
            all_entity_seqs.append(entity_id_int)

    if len(all_x_seqs) == 0:
        raise ValueError(
            f"No valid sequences could be built with sequence_length={sequence_length}. "
            f"Ensure entities have at least {sequence_length} years of data."
        )

    # Stack into arrays
    x_sequences: NDArray[np.float64] = np.stack(all_x_seqs, axis=0)
    y_sequences: NDArray[np.int64] = np.array(all_y_seqs, dtype=np.int64)
    sequence_entity_ids: NDArray[np.int64] = np.array(all_entity_seqs, dtype=np.int64)

    return {
        "x_sequences": x_sequences,
        "y_sequences": y_sequences,
        "sequence_entity_ids": sequence_entity_ids,
        "n_sequences": len(all_x_seqs),
        "n_features": n_features,
        "sequence_length": sequence_length,
    }


def compute_features_per_step(n_features: int, sequence_length: int) -> int:
    """Compute the LSTM input size for flat features reshaped to sequences.

    Rounds up, because reshape_flat_to_pseudo_sequences zero-pads the feature
    vector out to a multiple of sequence_length. Anything that builds an LSTM
    for such data must size its input layer with this same value, or it builds
    a differently-shaped model than the one that was trained.

    Args:
        n_features: Number of flat features per sample.
        sequence_length: Number of timesteps to reshape into.

    Returns:
        Features per timestep, i.e. the LSTM's input_size.
    """
    return (n_features + sequence_length - 1) // sequence_length


def reshape_flat_to_pseudo_sequences(
    x_features: NDArray[np.float64],
    sequence_length: int,
) -> NDArray[np.float64]:
    """Reshape flat tabular data to pseudo-sequences for LSTM.

    When temporal entity/year information is not available, this function
    treats each feature as a timestep, creating sequences of shape
    (n_samples, seq_len, features_per_step).

    If n_features is not divisible by sequence_length, features are padded
    with zeros to make them divisible.

    Args:
        x_features: Feature matrix, shape (n_samples, n_features).
        sequence_length: Desired sequence length.

    Returns:
        Reshaped array of shape (n_samples, sequence_length, features_per_step).
    """
    n_samples: int = int(x_features.shape[0])
    n_features: int = int(x_features.shape[1])

    features_per_step: int = compute_features_per_step(n_features, sequence_length)
    total_needed: int = features_per_step * sequence_length

    # Pad if necessary
    if total_needed > n_features:
        pad_width: int = total_needed - n_features
        padding: NDArray[np.float64] = np.zeros((n_samples, pad_width), dtype=np.float64)
        x_padded: NDArray[np.float64] = np.concatenate([x_features, padding], axis=1)
    else:
        x_padded = x_features

    # Reshape to (n_samples, sequence_length, features_per_step)
    reshaped: NDArray[np.float64] = x_padded.reshape(n_samples, sequence_length, features_per_step)
    return reshaped


__all__ = [
    "SequenceData",
    "build_sequences",
    "compute_features_per_step",
    "reshape_flat_to_pseudo_sequences",
]
