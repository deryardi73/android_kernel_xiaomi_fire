/* SPDX-License-Identifier: Dual BSD/GPL */
/*
 * LZ4KDR: a speed-tuned derivative of Huawei's LZ4KD.
 *
 * Original LZ4KD:
 *   Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *   "LZ4K compression algorithm with delta compression"
 *
 * This derivative keeps LZ4KD's on-wire format (the compressed bitstream
 * is byte-for-byte compatible with the original decoder) but changes the
 * encoder's internal search behavior for speed, per a small set of
 * independently-benchmarked optimizations (see the "lz4kdr" userspace
 * benchmark suite this was validated against, and
 * lz4kd_extreme_optimization_ideas.md for the full idea list this was
 * drawn from). It is renamed from lz4kd to lz4kdr specifically so it is
 * not mistaken for an unmodified import of Huawei's original code -- all
 * credit for the underlying algorithm and format belongs to them.
 */

#ifndef _LZ4KDR_H
#define _LZ4KDR_H

typedef enum {
	LZ4KDR_STATUS_INCOMPRESSIBLE =  0, /* data did not compress */
	LZ4KDR_STATUS_FAILED         = -1, /* general failure */
	LZ4KDR_STATUS_READ_ERROR     = -2, /* corrupt/truncated input */
	LZ4KDR_STATUS_WRITE_ERROR    = -3
} lz4kdr_status;

/*
 * lz4kdr_encode_state_bytes_min() returns the number of bytes the
 * `state` scratch buffer passed to lz4kdr_encode() must be at least.
 */
unsigned lz4kdr_encode_state_bytes_min(void);

/*
 * lz4kdr_encode() compresses one 4KB-or-smaller input block.
 *
 * Return value:
 *   > 0  success; the size of the encoded data, always <= out_max.
 *   0    data did not compress (encoded size would meet or exceed
 *        out_limit, or wouldn't fit out_max at all).
 *   < 0  failure (bad arguments, or in_max > 4096).
 *
 * `state` must be >= lz4kdr_encode_state_bytes_min() bytes and must be
 * zeroed once by the caller before its first use (e.g. right after
 * allocation) -- lz4kdr does NOT re-zero it on every call, unlike
 * upstream LZ4KD. See lz4kdr_encode.c's probe-loop comment for why
 * that's still always correct: it never reads a byte it hasn't proven
 * is in-bounds and behind the current search position.
 *
 * `out_limit`, if nonzero and less than min(in_max, out_max), lets the
 * encoder bail out early once it's clear the result won't beat that
 * size (e.g. pass the backing store's "not worth compressing further
 * than this" threshold). 0 means "use min(in_max, out_max)".
 */
int lz4kdr_encode(
	void *const state,
	const void *const in,
	void *out,
	unsigned in_max,
	unsigned out_max,
	unsigned out_limit);

/*
 * lz4kdr_decode() reverses lz4kdr_encode() (and unmodified upstream
 * LZ4KD's encoder -- the format didn't change).
 *
 * Returns the decoded size (== in_max passed to the matching encode
 * call) on success, or a negative lz4kdr_status on failure.
 */
int lz4kdr_decode(
	const void *const in,
	void *const out,
	unsigned in_max,
	unsigned out_max);

#endif /* _LZ4KDR_H */
