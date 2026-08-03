# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Tests for CP-aware sequence padding in materialize_bin_arrays."""

from __future__ import annotations

import numpy as np

from nemotron.data_prep.packing.bin_assignment import BinAssignment
from nemotron.data_prep.packing.materialize import _padded_seq_len, materialize_bin_arrays
from nemotron.data_prep.packing.spool import SequenceSpoolPaths, SequenceSpoolWriter


class _LocalSpool:
    def __init__(self, root: str) -> None:
        import fsspec

        self.fs = fsspec.filesystem("file")
        self.paths = SequenceSpoolPaths.for_root(root)
        writer = SequenceSpoolWriter(fs=self.fs, paths=self.paths)
        writer.append([10, 11, 12, 13, 14], [1, 1, 1, 1, 1])  # len 5 -> ceil(8)+1 = 9
        writer.append([20, 21], [1, 1])  # len 2 -> ceil(8)+1 = 9
        writer.finalize()

    def reader(self):
        from nemotron.data_prep.packing.spool import SequenceSpoolReader

        return SequenceSpoolReader(fs=self.fs, paths=self.paths)


def test_padded_seq_len_rounds_up(tmp_path) -> None:
    # CP alignment: ceil to a multiple of pad_seq_to_mult, then +1 so that
    # (stored_len - 1) is a multiple of the target (Megatron-Bridge drops the
    # last token of each sub-sequence when building cu_seqlens).
    assert _padded_seq_len(5, 8) == 9
    assert _padded_seq_len(8, 8) == 9
    assert _padded_seq_len(9, 8) == 17
    assert _padded_seq_len(5, None) == 5
    assert _padded_seq_len(5, 1) == 5
    # The load-bearing invariant for THD context parallelism.
    for s in range(1, 40):
        assert (_padded_seq_len(s, 8) - 1) % 8 == 0


def test_materialize_bin_arrays_cp_padding(tmp_path) -> None:
    spool = _LocalSpool(str(tmp_path / "spool"))
    reader = spool.reader()
    assignment = BinAssignment.from_bins(bins=[[0, 1]], num_sequences=2)

    scratch_input_ids = np.zeros((32,), dtype=np.int32)
    scratch_loss_mask = np.zeros((32,), dtype=np.uint8)

    packed_len, seq_start_id = materialize_bin_arrays(
        spool_reader=reader,
        assignment=assignment,
        bin_id=0,
        pack_size=32,
        scratch_input_ids=scratch_input_ids,
        scratch_loss_mask=scratch_loss_mask,
        pad_seq_to_mult=8,
    )

    # Each of the two sequences occupies ceil(len, 8) + 1 = 9 slots.
    assert packed_len == 18
    assert seq_start_id.tolist() == [0, 9]
    assert (scratch_input_ids[:5] == [10, 11, 12, 13, 14]).all()
    assert (scratch_input_ids[5:9] == 0).all()
    assert (scratch_input_ids[9:11] == [20, 21]).all()
    assert (scratch_input_ids[11:18] == 0).all()

    # The load-bearing invariant: Megatron-Bridge builds cu_seqlens with
    # (span - 1) per sub-sequence, and THD context parallelism requires each of
    # those to be a multiple of pad_seq_to_mult.
    boundaries = list(seq_start_id) + [packed_len]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        assert (end - start - 1) % 8 == 0
