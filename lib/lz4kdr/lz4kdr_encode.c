// SPDX-License-Identifier: Dual BSD/GPL
/*
 * LZ4KDR -- speed-tuned derivative of Huawei LZ4KD. See include/linux/
 * lz4kdr.h for the attribution note and lz4kdr_decode.c's header for
 * why decode is untouched.
 *
 * This encoder carries five changes vs. upstream LZ4KD, each
 * independently measured positive across four synthetic benchmark
 * corpora (zero/high/medium/low compressibility) AND three real-world
 * corpora (a compiled executable, an already-compressed video, real
 * prose/markup text) before being combined here -- see the "lz4kdr"
 * userspace benchmark suite this was validated against and
 * lz4kd_extreme_optimization_ideas.md for the full writeup. Every
 * change is called out inline where it occurs:
 *
 *   1. No per-call memset() of the hash table; a branchless q<r guard
 *      in the probe loop takes over the safety argument instead.
 *   2. HT_LOG2 12->10: 8KB->2KB hash table (a real CR/speed trade,
 *      the only one of these changes with a downside -- see the
 *      comment on HT_LOG2 below).
 *   3. Adaptive search-step acceleration, now backed by a genuine
 *      per-CPU variable (this needed a real kernel home; the
 *      benchmark's file-scope static was an acknowledged stand-in).
 *   4. One 8-byte read feeds both of the probe loop's hash lookups
 *      per iteration instead of two separate reads.
 *   5. Branchless tag-escape-field selection in the two hot tag-write
 *      paths (update_utag(), out_repeat()).
 *
 * Changes 4 and 5 were originally measured as isolated net LOSSES
 * against plain upstream LZ4KD (dual-hash: -2.2%/-3.5% on zero/high;
 * branchless tag: -2.0%/-5.7%) and excluded from the first release of
 * this encoder (1.0-1.2, changes 1-3 only). Re-testing them on TOP of
 * changes 1-3 instead of plain upstream found they flip to real wins
 * on zero/high (+7.8-8.5% each, +10-11% stacked) with no new
 * regression elsewhere: changes 1-3 already sped up everything else
 * in the probe loop and tag-write path on long-run-heavy data, so the
 * same fixed per-call cost these two changes remove became a bigger
 * share of a now-smaller total runtime. This is the same bottleneck-
 * shift mechanism documented in the userspace suite's RESULTS.md
 * ("Bottleneck-shift synergy re-test" section) -- see it for the full
 * paired-benchmark methodology and numbers, including why a naive
 * single-shot comparison initially looked like a much bigger (and
 * wrong) win before controlling for CPU turbo/warm-up drift.
 *
 * An AVX2 32B vectorized match-length scan in repeat_end() -- and,
 * separately, a hardware-CRC32 hash (superseded outright by change 4,
 * which removes the only call site CRC32 would have sped up) -- were
 * also implemented, benchmarked, and REMOVED. The AVX2 scan was a
 * huge win on the synthetic zero/high corpora (+40-54%, +61-90% on
 * top of changes 1-5) but a real 7-24% REGRESSION on real compiled-
 * executable data in every combination tested: matches in real
 * machine code rarely run past a handful of bytes, so the vector
 * path's fixed load+compare+movemask cost gets paid on nearly every
 * match found and essentially never amortized -- wider vectors made
 * this *worse* (AVX-512 regressed executables more than AVX2, which
 * regressed them more than SSE2). Two rounds of heuristic gating (an
 * 8-byte peek before committing to the vector loop, then a page-level
 * adaptive on/off streak mirroring change 3's mechanism) only
 * partially closed the gap, down to a residual ~8-10% regression at
 * best, and the page-level version underperformed the simpler
 * per-match peek. Rather than ship a partial fix for a real
 * regression on an extremely common real-world zram content type
 * (compiled code -- shared libraries, executables), it was dropped
 * entirely. If revisiting this, the interesting open question is
 * whether a per-block cheap classifier decided *before* compression
 * starts (rather than gating each individual match) could tell long-
 * match-heavy pages (worth vectorizing) apart from short-match-heavy
 * ones (not) accurately enough to be worth the complexity.
 */

#if !defined(__KERNEL__)
#include "lz4kdr.h"
#else
#include <linux/kernel.h>
#include <linux/lz4kdr.h>
#include <linux/module.h>
#include <linux/percpu.h>
#endif

#include "lz4kdr_private.h"
#include "lz4kdr_encode_private.h"
#include "version.h"

enum {
	/*
	 * change 2: "turbo" hash table. Upstream LZ4KD used 12 (4096
	 * entries, 8KB). 10 -> 1024 entries, 2KB, comfortably L1d-resident
	 * on essentially any core -- at the cost of more hash collisions,
	 * i.e. a real (measured ~2.5% on realistic code/text data)
	 * compression-ratio regression. This is the only one of the four
	 * changes with a downside; if that trade isn't acceptable for a
	 * given deployment, this is the one line to revert back to 12.
	 */
	HT_LOG2 = 10,
	STEP_LOG2 = 5 /* ==3 #2 avg drop in CR */
};

/*
 * change 3: adaptive search-step acceleration state.
 *
 * zcomp_stream_get() disables preemption for the duration of a
 * compress call (it's a get_cpu_ptr() under the hood), so a plain
 * per-CPU variable is exactly the right home for this -- no locking,
 * no risk of one CPU's streak biasing another's, and it survives
 * across calls on the same CPU exactly like the per-CPU hash table
 * context does.
 */
enum { ACCEL_BIAS_MAX = 3 };

#if defined(__KERNEL__)
static DEFINE_PER_CPU(unsigned int, lz4kdr_accel_bias);
#define ACCEL_BIAS_GET()   this_cpu_read(lz4kdr_accel_bias)
#define ACCEL_BIAS_INC()   this_cpu_inc(lz4kdr_accel_bias)
#define ACCEL_BIAS_RESET() this_cpu_write(lz4kdr_accel_bias, 0)
#else
static unsigned int g_accel_bias;
#define ACCEL_BIAS_GET()   (g_accel_bias)
#define ACCEL_BIAS_INC()   (g_accel_bias++)
#define ACCEL_BIAS_RESET() (g_accel_bias = 0)
#endif

static unsigned encode_state_bytes_min(void)
{
	enum {
		BYTES_LOG2 = HT_LOG2 + 1
	};
	const unsigned bytes_total = (1U << BYTES_LOG2);
	return bytes_total;
}

#if !defined(LZ4K_DELTA) && !defined(LZ4K_MAX_CR)

unsigned lz4kdr_encode_state_bytes_min(void)
{
	return encode_state_bytes_min();
}
EXPORT_SYMBOL(lz4kdr_encode_state_bytes_min);

#endif /* !defined(LZ4K_DELTA) && !defined(LZ4K_MAX_CR) */

/* minimum encoded size for non-compressible data */
inline static uint_fast32_t encoded_bytes_min(
	uint_fast32_t nr_log2,
	uint_fast32_t in_max)
{
	return in_max < mask(nr_log2) ?
		TAG_BYTES_MAX + in_max :
		TAG_BYTES_MAX + size_bytes_count(in_max - mask(nr_log2)) + in_max;
}

/*
 * change 5: branchless escape-field selection. `cond ? a : b` on these
 * hot tag-write paths becomes a 0/-1 select mask combined with &/| --
 * see update_utag() and out_repeat() below, and the top-of-file
 * comment for why this flips positive once changes 1-3 have already
 * sped up everything else around it (isolated against plain upstream,
 * it's a net loss: the branch is usually well-predicted since escapes
 * are rare, and unconditionally computing both sides costs more than
 * that saves).
 */
inline static uint_fast32_t branchless_select(bool cond, uint_fast32_t a, uint_fast32_t b)
{
	const uint_fast32_t m = (uint_fast32_t)0 - (uint_fast32_t)cond;
	return (a & m) | (b & ~m);
}

inline static void  update_utag(
	uint_fast32_t r_bytes_max,
	uint_fast32_t *utag,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	const uint_fast32_t r_mask = mask(TAG_BITS_MAX - (off_log2 + nr_log2));
	*utag |= branchless_select(r_bytes_max - REPEAT_MIN < r_mask,
				    (r_bytes_max - REPEAT_MIN) << off_log2,
				    r_mask << off_log2);
}

inline static uint8_t *out_size_bytes(uint8_t *out_at, uint_fast32_t u)
{
	for (; u >= BYTE_MAX; *out_at++ = (uint8_t)BYTE_MAX, u -= BYTE_MAX);
	*out_at++ = (uint8_t)u;
	return out_at;
}

inline static uint8_t *out_utag_then_bytes_left(
	uint8_t *out_at,
	uint_fast32_t utag,
	uint_fast32_t bytes_left)
{
	m_copy(out_at, &utag, TAG_BYTES_MAX);
	return out_size_bytes(out_at + TAG_BYTES_MAX, bytes_left);
}

static int out_tail(
	uint8_t *out_at,
	uint8_t *const out_end,
	const uint8_t *const out,
	const uint8_t *const nr0,
	const uint8_t *const in_end,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	const uint_fast32_t nr_mask = mask(nr_log2);
	const uint_fast32_t r_log2 = TAG_BITS_MAX - (off_log2 + nr_log2);
	const uint_fast32_t nr_bytes_now = u_32(in_end - nr0);
	if (encoded_bytes_min(nr_log2, nr_bytes_now) > u_32(out_end - out_at))
		return LZ4KDR_STATUS_INCOMPRESSIBLE;
	if (nr_bytes_now < nr_mask) {
		/* caller guarantees at least one nr-byte */
		uint_fast32_t utag = (nr_bytes_now << (off_log2 + r_log2));
		m_copy(out_at, &utag, TAG_BYTES_MAX);
		out_at += TAG_BYTES_MAX;
	} else { /* nr_bytes_now>=nr_mask */
		uint_fast32_t bytes_left = nr_bytes_now - nr_mask;
		uint_fast32_t utag = (nr_mask << (off_log2 + r_log2));
		out_at = out_utag_then_bytes_left(out_at, utag, bytes_left);
	} /* if (nr_bytes_now<nr_mask) */
	m_copy(out_at, nr0, nr_bytes_now);
	return (int)(out_at + nr_bytes_now - out);
}

inline static int out_tail2(
	uint8_t *out_at,
	uint8_t *const out_end,
	const uint8_t *const out,
	const uint8_t *const r,
	const uint8_t *const in_end,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	return r == in_end ? (int)(out_at - out) :
		out_tail(out_at, out_end, out, r, in_end,
			 nr_log2, off_log2);
}

int lz4kdr_out_tail(
	uint8_t *out_at,
	uint8_t *const out_end,
	const uint8_t *const out,
	const uint8_t *const nr0,
	const uint8_t *const in_end,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2,
	bool check_out)
{
	return out_tail(out_at, out_end, out, nr0, in_end,
			nr_log2, off_log2);
}

static uint8_t *out_non_repeat(
	uint8_t *out_at,
	uint8_t *const out_end,
	uint_fast32_t utag,
	const uint8_t *const nr0,
	const uint8_t *const r,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	const uint_fast32_t nr_bytes_max = u_32(r - nr0);
	const uint_fast32_t nr_mask = mask(nr_log2),
		r_log2 = TAG_BITS_MAX - (off_log2 + nr_log2);
	if (likely(nr_bytes_max < nr_mask)) {
		utag |= (nr_bytes_max << (off_log2 + r_log2));
		m_copy(out_at, &utag, TAG_BYTES_MAX);
		out_at += TAG_BYTES_MAX;
	} else { /* nr_bytes_max >= nr_mask */
		uint_fast32_t bytes_left = nr_bytes_max - nr_mask;
		utag |= (nr_mask << (off_log2 + r_log2));
		out_at = out_utag_then_bytes_left(out_at, utag, bytes_left);
	} /* if (nr_bytes_max<nr_mask) */
	copy_x_while_total(out_at, nr0, nr_bytes_max, NR_COPY_MIN);
	out_at += nr_bytes_max;
	return out_at;
}

inline static uint8_t *out_r_bytes_left(
	uint8_t *out_at,
	uint_fast32_t r_bytes_max,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	const uint_fast32_t r_mask = mask(TAG_BITS_MAX - (off_log2 + nr_log2));
	return likely(r_bytes_max - REPEAT_MIN < r_mask) ?
		out_at : out_size_bytes(out_at, r_bytes_max - REPEAT_MIN - r_mask);
}

static uint8_t *out_repeat(
	uint8_t *out_at,
	uint_fast32_t utag,
	uint_fast32_t r_bytes_max,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	const uint_fast32_t r_mask = mask(TAG_BITS_MAX - (off_log2 + nr_log2));
	const bool no_escape = r_bytes_max - REPEAT_MIN < r_mask;
	utag |= branchless_select(no_escape,
				   (r_bytes_max - REPEAT_MIN) << off_log2,
				   r_mask << off_log2);
	m_copy(out_at, &utag, TAG_BYTES_MAX);
	out_at += TAG_BYTES_MAX;
	if (unlikely(!no_escape)) {
		uint_fast32_t bytes_left = r_bytes_max - REPEAT_MIN - r_mask;
		out_at = out_size_bytes(out_at, bytes_left);
	}
	return out_at; /* SUCCESS: continue compression */
}

uint8_t *lz4kdr_out_repeat(
	uint8_t *out_at,
	uint8_t *const out_end,
	uint_fast32_t utag,
	uint_fast32_t r_bytes_max,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2,
	const bool check_out)
{
	return out_repeat(out_at, utag, r_bytes_max, nr_log2, off_log2);
}

inline static uint8_t *out_tuple(
	uint8_t *out_at,
	uint8_t *const out_end,
	uint_fast32_t utag,
	const uint8_t *const nr0,
	const uint8_t *const r,
	uint_fast32_t r_bytes_max,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2)
{
	update_utag(r_bytes_max, &utag, nr_log2, off_log2);
	out_at = out_non_repeat(out_at, out_end, utag, nr0, r, nr_log2, off_log2);
	return out_r_bytes_left(out_at, r_bytes_max, nr_log2, off_log2);
}

uint8_t *lz4kdr_out_tuple(
	uint8_t *out_at,
	uint8_t *const out_end,
	uint_fast32_t utag,
	const uint8_t *const nr0,
	const uint8_t *const r,
	uint_fast32_t r_bytes_max,
	const uint_fast32_t nr_log2,
	const uint_fast32_t off_log2,
	bool check_out)
{
	return out_tuple(out_at, out_end, utag, nr0, r, r_bytes_max,
				nr_log2, off_log2);
}

static const uint8_t *repeat_end(
	const uint8_t *q,
	const uint8_t *r,
	const uint8_t *const in_end_safe,
	const uint8_t *const in_end)
{
	q += REPEAT_MIN;
	r += REPEAT_MIN;
	/* An AVX2 32B-at-a-time vector compare lived here; removed after
	 * real-world benchmarking found it regressed compiled-executable
	 * data 7-24% (see the top-of-file comment) -- back to upstream
	 * LZ4KD's original scalar 8B loop, unmodified. */
	do {
		const uint64_t x = read8_at(q) ^ read8_at(r);
		if (x) {
			const uint16_t ctz = (uint16_t)__builtin_ctzl(x);
			return r + (ctz >> BYTE_BITS_LOG2);
		}
		/* some bytes differ: count of trailing 0-bits/bytes */
		q += sizeof(uint64_t);
		r += sizeof(uint64_t);
	} while (likely(r <= in_end_safe)); /* once, at input block end */
	while (r < in_end) {
		if (*q != *r) return r;
		++q;
		++r;
	}
	return r;
}

const uint8_t *lz4kdr_repeat_end(
	const uint8_t *q,
	const uint8_t *r,
	const uint8_t *const in_end_safe,
	const uint8_t *const in_end)
{
	return repeat_end(q, r, in_end_safe, in_end);
}

/* CR increase order: +STEP, have OFFSETS, use _5b(most impact) */
/* *_6b to compete with LZ4 */
inline static uint_fast32_t hash(const uint8_t *r)
{
	return hash64_5b(r, HT_LOG2);
}

/*
 * Proof that 'r' increments are safe-NO pointer overflows are possible:
 *
 * While using STEP_LOG2=5, step_start=1<<STEP_LOG2 == 32 we increment s
 * 32 times by 1, 32 times by 2, 32 times by 3, and so on:
 * 32*1+32*2+32*3+...+32*31 == 32*SUM(1..31) == 32*((1+31)*15+16).
 * So, we can safely increment s by at most 31 for input block size <=
 * 1<<13 < 15872.
 *
 * More precisely, STEP_LIMIT == x for any input block  calculated as follows:
 * 1<<off_log2 >= (1<<STEP_LOG2)*((x+1)(x-1)/2+x/2) ==>
 * 1<<(off_log2-STEP_LOG2+1) >= x^2+x-1 ==>
 * x^2+x-1-1<<(off_log2-STEP_LOG2+1) == 0, which is solved by standard
 * method.
 * To avoid overhead here conservative approximate value of x is calculated
 * as average of two nearest square roots, see STEP_LIMIT above.
 *
 * NOTE (change 3): step's *initial* value is additionally left-shifted
 * by the adaptive accel bias below -- it stays within the same growth
 * proof above since biasing only ever makes the effective STEP_LOG2
 * larger (searches skip further per step, not more of them).
 */

static int encode_any(
	uint16_t *const ht,
	const uint8_t *const in0,
	const uint8_t *const in_end,
	uint8_t *const out,
	uint8_t *const out_end)
{
	enum {
		NR_LOG2 = NR_4KB_LOG2,
		OFF_LOG2 = BLOCK_4KB_LOG2
	};
	const uint8_t *const in_end_safe = in_end - NR_COPY_MIN;
	const uint8_t *r = in0;
	const uint8_t *nr0 = r++;
	uint8_t *out_at = out + 1; /* +1 for header */
	for (; ; nr0 = r) {
		const uint8_t *q = 0;
		/* change 3: bias the initial search step from recent pages'
		 * compressibility outcome (see lz4kdr_encode()'s tail). */
		uint_fast32_t step = (1 << STEP_LOG2) << ACCEL_BIAS_GET();
		uint_fast32_t utag = 0;
		const uint8_t *r_end = 0;
		uint_fast32_t r_bytes_max = 0;
		while (true) {
			/*
			 * change 1: q<r guard against stale ht[] entries.
			 * `state` is no longer memset per-call (see
			 * lz4kdr_encode() below); a hash bucket may still
			 * hold an offset left over from a PAST call. That
			 * offset is always < BLOCK_4KB (a uint16_t written
			 * only as r-in0, with in_max capped at 4096), so
			 * q=in0+ht[h] can never read out of bounds -- but it
			 * could point AHEAD of the current r (q>=r), which
			 * would make utag=u_32(r-q) wrap and corrupt the
			 * 12-bit offset field written into the compressed
			 * stream. The q<r guard rejects exactly that case;
			 * anything that also passes equal4()/equal4pv() is a
			 * real, correct match against live buffer content --
			 * this doesn't depend on the table ever having been
			 * cleared, only on this one inequality. (Combined
			 * via `&`, not `&&`: the naive short-circuit form
			 * compiles to two separately-predicted branches, and
			 * q<r's outcome is NOT well-predicted once the table
			 * isn't reset every call -- that cost more in
			 * mispredictions than the memset saved on non-trivial
			 * data. `&` forces a single, well-predicted branch,
			 * same branch count as upstream had.) This guard
			 * applies to both of change 4's probes below, same as
			 * it did to the two separate hashed() calls it
			 * replaced.
			 */
			/*
			 * change 4: one 8-byte read feeds BOTH this
			 * iteration's hash lookups. hash64_5b/hash64v_5b only
			 * consume the low 5 bytes of their input (via <<24),
			 * so v's low 5 bytes = r[0..5) hash position r, and
			 * (v>>8)'s low 5 bytes = r[1..6) hash position r+1 --
			 * one load instead of two, and both ht[] loads/hashes
			 * can issue back to back instead of waiting on the
			 * first equal4pv(). This bypasses hash()/hashed()
			 * entirely for the probe loop; hash() is still used
			 * once per match below (ht[hash(r - 1)]), just far
			 * less often than it was called here.
			 */
			{
				const uint64_t v = read8_at(r);
				const uint_fast32_t h0 = hash64v_5b(v, HT_LOG2);
				q = in0 + ht[h0];
				ht[h0] = (uint16_t)(r - in0);
				if ((q < r) & equal4pv(q, v))
					break;
				++r;
				{
					const uint64_t v1 = v >> 8;
					const uint_fast32_t h1 = hash64v_5b(v1, HT_LOG2);
					q = in0 + ht[h1];
					ht[h1] = (uint16_t)(r - in0);
					if ((q < r) & equal4pv(q, v1))
						break;
				}
			}
			if (unlikely((r += (++step >> STEP_LOG2)) > in_end_safe))
				return out_tail(out_at, out_end, out, nr0, in_end,
						NR_LOG2, OFF_LOG2);
		}
		utag = u_32(r - q);
		r_end = repeat_end(q, r, in_end_safe, in_end);
		r_bytes_max = u_32(r_end - r);
		if (unlikely(nr0 == r))
			out_at = out_repeat(out_at, utag, r_bytes_max,
					    NR_LOG2, OFF_LOG2);
		else
			out_at = out_tuple(out_at, out_end, utag, nr0, r, r_bytes_max,
					    NR_LOG2, OFF_LOG2);
		if (unlikely((r += r_bytes_max) > in_end_safe))
			return out_tail2(out_at, out_end, out, r, in_end,
					 NR_LOG2, OFF_LOG2);
		ht[hash(r - 1)] = (uint16_t)(r - 1 - in0);
	}
}

/* not static for inlining optimization */
int lz4kdr_encode_fast(
	void *const state,
	const uint8_t *const in,
	uint8_t *const out,
	const uint_fast32_t in_max,
	const uint_fast32_t out_max)
{
	return encode_any((uint16_t*)state, in, in + in_max, out, out + out_max);
}

int lz4kdr_encode(
	void *const state,
	const void *const in,
	void *out,
	unsigned in_max,
	unsigned out_max,
	unsigned out_limit)
{
	const uint64_t io_min = min_u64(in_max, out_max);
	const uint64_t gain_max = max_u64(GAIN_BYTES_MAX, (io_min >> GAIN_BYTES_LOG2));
	/* ++use volatile pointers to prevent compiler optimizations */
	const uint8_t *volatile in_end = (const uint8_t*)in + in_max;
	const uint8_t *volatile out_end = (uint8_t*)out + out_max;
	const void *volatile state_end =
		(uint8_t*)state + encode_state_bytes_min();
	if (unlikely(state == NULL))
		return LZ4KDR_STATUS_FAILED;
	if (unlikely(in == NULL || out == NULL))
		return LZ4KDR_STATUS_FAILED;
	if (unlikely(out_max <= gain_max))
		return LZ4KDR_STATUS_FAILED;
	if (unlikely((const uint8_t*)in >= in_end || (uint8_t*)out >= out_end))
		return LZ4KDR_STATUS_FAILED;
	if (unlikely(state >= state_end))
		return LZ4KDR_STATUS_FAILED; /* pointer overflow */
	if (in_max > (1 << BLOCK_4KB_LOG2))
		return LZ4KDR_STATUS_FAILED;
	if (unlikely(!out_limit || out_limit > io_min))
		out_limit = (unsigned)io_min;
	if (unlikely(nr_encoded_bytes_max(in_max, NR_4KB_LOG2) > out_max))
		return 0;
	/* change 1: memset(state) removed -- safety now rests on the q<r
	 * guard in encode_any()'s probe loop, above. The caller must still
	 * zero `state` once at allocation time (cold start); see
	 * backend_lz4kdr.c's create_ctx, which kzalloc()s it. */
	*((uint8_t*)out) = 0; /* lz4kdr header */
	{
		int ret;
		ret = lz4kdr_encode_fast(state, (const uint8_t*)in,
				(uint8_t*)out, in_max, out_limit);
		/* change 3: update the acceleration bias from this page's
		 * outcome. Anything <=0 or under ~12% shrinkage counts as a
		 * "poor" page and nudges the bias up (cap ACCEL_BIAS_MAX);
		 * any decent compression resets it. */
		if (ret <= 0 || (uint_fast32_t)ret > (in_max - (in_max >> 3))) {
			if (ACCEL_BIAS_GET() < ACCEL_BIAS_MAX)
				ACCEL_BIAS_INC();
		} else {
			ACCEL_BIAS_RESET();
		}
		return ret;
	}
}
EXPORT_SYMBOL(lz4kdr_encode);

/* maximum encoded size for repeat and non-repeat data if "fast" encoder is used */
uint_fast32_t lz4kdr_encoded_bytes_max(
	uint_fast32_t nr_max,
	uint_fast32_t r_max,
	uint_fast32_t nr_log2,
	uint_fast32_t off_log2)
{
	uint_fast32_t r = 1 + TAG_BYTES_MAX +
		(uint32_t)round_up_to_log2(nr_max, NR_COPY_LOG2);
	uint_fast32_t r_log2 = TAG_BITS_MAX - (off_log2 + nr_log2);
	if (nr_max >= mask(nr_log2))
		r += size_bytes_count(nr_max - mask(nr_log2));
	if (r_max >= mask(r_log2)) {
		r_max -= mask(r_log2);
		r += (uint_fast32_t)max_u64(size_bytes_count(r_max),
					r_max - r_max / REPEAT_MIN); /* worst case: one tag for each REPEAT_MIN */
	}
	return r;
}
EXPORT_SYMBOL(lz4kdr_encoded_bytes_max);

static int __init lz4kdr_init(void)
{
	printk(KERN_INFO "%s %s by %s\n",
		LZ4KDR_PROGNAME, LZ4KDR_VERSION, LZ4KDR_AUTHOR);
	return 0;
}
module_init(lz4kdr_init);

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR(LZ4KDR_AUTHOR);
MODULE_DESCRIPTION(LZ4KDR_PROGNAME);
