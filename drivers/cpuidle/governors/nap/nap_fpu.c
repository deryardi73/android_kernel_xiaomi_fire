#include <linux/cpuidle.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/pm_qos.h>
#include <linux/sched/clock.h>
#include <linux/string.h>
#include <linux/tick.h>

#include "nap.h"

typedef float v4sf __attribute__((__vector_size__(16)));
typedef int   v4si __attribute__((__vector_size__(16)));
typedef unsigned int v4ui __attribute__((__vector_size__(16)));

static inline float float_min(float a, float b) { return a < b ? a : b; }
static inline float float_max(float a, float b) { return a > b ? a : b; }

static inline v4sf v4sf_min(v4sf a, v4sf b)
{
	asm("fmin %0.4s, %1.4s, %2.4s" : "=w"(a) : "w"(a), "w"(b));
	return a;
}

static inline v4sf v4sf_max(v4sf a, v4sf b)
{
	asm("fmax %0.4s, %1.4s, %2.4s" : "=w"(a) : "w"(a), "w"(b));
	return a;
}

static inline float nap_sqrtf(float x)
{
	float r;
	asm("fsqrt %s0, %s1" : "=w"(r) : "w"(x));
	return r;
}

static inline float fast_log2f(float x)
{
	union { float f; u32 i; } u = { .f = x };
	int exp = (int)((u.i >> 23) & 0xFFu) - 127;
	float e = (float)exp;
	float m, p;

	u.i = (u.i & 0x7FFFFFu) | (127u << 23);
	m = u.f - 1.0f;

	p = m * 0.4808f;
	p = 0.7213f - p;
	p = m * p;
	p = 1.4425f - p;
	p = m * p;

	return e + p;
}

static inline float fast_exp2f(float x)
{
	union { u32 i; float f; } v;
	int xi;
	float f;

	if (x > 60.0f) x = 60.0f;
	else if (x < -60.0f) x = -60.0f;

	xi = (int)x;
	if (x < (float)xi) xi--;
	f = x - (float)xi;

	v.i = (u32)((xi + 127) << 23);
	return v.f * (1.0f + f * (0.6931472f +
			f * (0.2402265f + f * 0.0555041f)));
}

static inline float nap_sigmoidf(float x)
{
	return 1.0f / (1.0f + fast_exp2f(-1.4426950f * x));
}

#define NAP_FLOOR_WIN  256
#define NAP_PRIOR_K    16

static inline float nap_prng_float(u32 *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)(s32)*state * (1.0f / 2147483648.0f);
}

static inline void nap_nn_forward(const float *input, float *output,
				  float *hidden_save,
				  const struct nap_weights *w)
{
	nap_nn_forward_neon(input, output, hidden_save, w);
}

static inline void nap_nn_learn(struct nap_cpu_data *d)
{
	nap_nn_learn_neon(d);
}

#define NAP_PRNG_SEED 42u

static void nap_init_weights(struct nap_weights *w)
{
	u32 rng = NAP_PRNG_SEED;
	float scale_h1, scale_out;
	int i, j;

	scale_h1  = nap_sqrtf(6.0f / (float)(NAP_INPUT_SIZE + NAP_HIDDEN_SIZE));
	scale_out = 0.01f;

	for (i = 0; i < NAP_INPUT_SIZE; i++)
		for (j = 0; j < NAP_HIDDEN_SIZE; j++)
			w->w_h1[i][j] = nap_prng_float(&rng) * scale_h1;

	memset(w->b_h1, 0, sizeof(w->b_h1));

	for (j = 0; j < NAP_HIDDEN_SIZE; j++)
		w->w_out[j] = nap_prng_float(&rng) * scale_out;

	w->b_out = 0.0f;

	for (i = 0; i < NAP_INPUT_SIZE; i++)
		w->w_h1[i][0] = 0.0f;
	w->w_h1[0][0] = 1.0f;
	w->b_h1[0] = 0.0f;
	w->w_out[0] = 1.0f;
}

static void nap_init_log2_tres(struct nap_cpu_data *d,
			       struct cpuidle_driver *drv)
{
	int i;

	for (i = 0; i < drv->state_count; i++) {
		float tres = float_max(
			(float)drv->states[i].target_residency * NSEC_PER_USEC, 1.0f);
		d->log2_tres[i] = fast_log2f(tres);
	}

	for (i = 1; i < drv->state_count; i++)
		d->weights.thr_ord[i - 1] = d->log2_tres[i];
}

struct logring_stats {
	float avg;
	float min;
	float max;
};

static inline v4sf fast_log2f_vec(v4sf x)
{
	v4sf m, p;
	v4ui bits;
	v4si exp;
	union { v4sf f; v4ui i; } u;

	u.f = x;
	bits = u.i;
	exp = (v4si)(bits >> 23) - (v4si){ 127, 127, 127, 127 };
	bits = (bits & (v4ui){ 0x7FFFFF, 0x7FFFFF, 0x7FFFFF, 0x7FFFFF })
		| (v4ui){ 127 << 23, 127 << 23, 127 << 23, 127 << 23 };
	__builtin_memcpy(&m, &bits, sizeof(m));
	m = m - (v4sf){ 1.0f, 1.0f, 1.0f, 1.0f };

	p = m * (v4sf){ 0.4808f, 0.4808f, 0.4808f, 0.4808f };
	p = (v4sf){ 0.7213f, 0.7213f, 0.7213f, 0.7213f } - p;
	p = m * p;
	p = (v4sf){ 1.4425f, 1.4425f, 1.4425f, 1.4425f } - p;
	p = m * p;

	return (v4sf)__builtin_convertvector(exp, v4sf) + p;
}

static void logring_compute(const struct nap_cpu_data *d,
			    struct logring_stats *s)
{
	int i, n = d->hist_count;
	float sum;

	if (n == 0) {
		*s = (struct logring_stats){ 0 };
		return;
	}

	if (n == NAP_HISTORY_SIZE) {
		v4sf v0, v1, pmin, pmax, psum;
		__builtin_memcpy(&v0, &d->log_history[0], sizeof(v0));
		__builtin_memcpy(&v1, &d->log_history[4], sizeof(v1));
		pmin = v4sf_min(v0, v1);
		pmax = v4sf_max(v0, v1);
		psum = v0 + v1;

		s->min = pmin[0];
		s->max = pmax[0];
		sum = psum[0];
		for (i = 1; i < 4; i++) {
			if (pmin[i] < s->min) s->min = pmin[i];
			if (pmax[i] > s->max) s->max = pmax[i];
			sum += psum[i];
		}
	} else {
		sum = d->log_history[0];
		s->min = sum;
		s->max = sum;

		for (i = 1; i < n; i++) {
			float val = d->log_history[i];
			sum += val;
			if (val < s->min) s->min = val;
			if (val > s->max) s->max = val;
		}
	}

	s->avg = sum / (float)n;
}

static void nap_extract_features(struct cpuidle_driver *drv,
				 struct cpuidle_device *dev,
				 float out[NAP_INPUT_SIZE],
				 s64 latency_req)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct logring_stats lr;
	ktime_t sleep_length, delta_tick;
	u64 busy_ns;
	float log_inputs[4];
	float log_results[4];

	sleep_length = tick_nohz_get_sleep_length(&delta_tick);
	busy_ns = local_clock() - d->prev_idle_exit;

	{
		float err_f = (float)(d->last_prediction_error / 1000);
		float abs_err = (err_f >= 0.0f) ? err_f : -err_f;

		log_inputs[0] = float_max((float)ktime_to_ns(sleep_length), 1.0f);
		log_inputs[1] = float_max((float)(dev->last_residency * NSEC_PER_USEC), 1.0f);
		log_inputs[2] = float_max((float)busy_ns, 1.0f);
		log_inputs[3] = abs_err + 1.0f;

		{
			v4sf log_in;
			v4sf log_out;
			__builtin_memcpy(&log_in, log_inputs, sizeof(log_in));
			log_out = fast_log2f_vec(log_in);
			__builtin_memcpy(log_results, &log_out, sizeof(log_results));
		}

		out[0] = log_results[0];
		out[1] = log_results[1];
		out[6] = log_results[2];

		{
			u32 sgn = ((u32 *)&err_f)[0] & 0x80000000u;
			u32 mag = ((u32 *)&log_results[3])[0] & 0x7FFFFFFFu;
			u32 res = mag | sgn;
			out[5] = *(float *)&res;
		}
	}

	{
		int prev = (d->hist_idx - 1 + NAP_HISTORY_SIZE) % NAP_HISTORY_SIZE;
		d->log_history[prev] = log_results[1];
	}

	logring_compute(d, &lr);
	out[2] = lr.avg;
	out[3] = lr.min;
	out[4] = lr.max;

	{
		u64 deepest_lat = (u64)drv->states[drv->state_count - 1]
				      .exit_latency * NSEC_PER_USEC;
		bool lat_valid = (latency_req < PM_QOS_LATENCY_ANY_NS &&
				  deepest_lat > 0);

		if (lat_valid)
			out[7] = fast_log2f(float_max((float)latency_req, 1.0f))
			       - fast_log2f(float_max((float)deepest_lat, 1.0f));
		else
			out[7] = 0.0f;
	}

	d->last_predicted_ns = ktime_to_ns(sleep_length);
}

int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d)
{
	s64 latency_req = (s64)cpuidle_governor_latency_req(dev->cpu) * NSEC_PER_USEC;

	if (unlikely(d->reset_pending)) {
		nap_init_weights(&d->weights);
		nap_init_log2_tres(d, drv);
		memset(d->bin_count, 0, sizeof(d->bin_count));
		memset(d->debt, 0, sizeof(d->debt));
		d->debt_decay_ns = 0;
		d->cooldown_until = 0;
		d->have_sample = false;
		d->stats.learn_count = 0;
		d->needs_learn = false;
		d->reset_pending = false;
	}

	if (d->have_sample) {
		float decay = (float)(NAP_FLOOR_WIN - 1) / (float)NAP_FLOOR_WIN;
		int k, label_bin = 0;

		if (d->needs_learn) {
			float base_lr = (float)d->learning_rate_millths / 1000.0f;
			float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
			float s = d->nn_output;
			float g = 0.0f;

			for (k = 1; k < drv->state_count; k++) {
				float th = d->active_w->thr_ord[k - 1];
				float q = nap_sigmoidf(s - th);
				float y = (d->learn_actual_ns >=
					   (u64)drv->states[k].target_residency * NSEC_PER_USEC)
					  ? 1.0f : 0.0f;
				float err = q - y;
				float lo = d->log2_tres[k] - 6.0f;
				float hi = d->log2_tres[k] + 6.0f;

				g += err;
				d->active_w->thr_ord[k - 1] =
					fclampf(th + fclampf(base_lr * err,
							     -clamp_val, clamp_val),
						lo, hi);
			}
			d->learn_d_out = g;
			d->learn_lr = base_lr;
			d->stats.learn_count++;
			nap_nn_learn(d);
			d->needs_learn = false;
		}

		for (k = 1; k < drv->state_count; k++)
			if (d->learn_actual_ns >=
			    (u64)drv->states[k].target_residency * NSEC_PER_USEC)
				label_bin = k;
		for (k = 0; k < drv->state_count; k++)
			d->bin_count[k] *= decay;
		d->bin_count[label_bin] += 1.0f;

		d->have_sample = false;
	}

	nap_extract_features(drv, dev, d->features_f32, latency_req);

	d->active_w = &d->weights;

	nap_nn_forward(d->features_f32, &d->nn_output, d->hidden_out,
		       d->active_w);

	{
		float conf = (float)d->conf_millths / 1000.0f;
		float s = d->nn_output;
		float sleep_log2 = d->features_f32[0];
		float suffix[CPUIDLE_STATE_MAX];
		float total = 0.0f;
		float qmin = 1.0f;
		int k, m = 0, idx = 0;
		u64 now_ns = local_clock();

		if (d->debt_decay_ns &&
		    now_ns - d->debt_decay_ns > NSEC_PER_MSEC * 100) {
			for (k = 0; k < drv->state_count; k++)
				d->debt[k] = (d->debt[k] > 1) ? (d->debt[k] / 2) : 0;
			d->debt_decay_ns = now_ns;
		}

		for (k = 0; k < drv->state_count; k++)
			total += d->bin_count[k];

		suffix[drv->state_count - 1] =
			d->bin_count[drv->state_count - 1];
		for (k = drv->state_count - 2; k >= 0; k--)
			suffix[k] = suffix[k + 1] + d->bin_count[k];

		for (k = 1; k < drv->state_count; k++) {
			float q_nn = nap_sigmoidf(s - d->active_w->thr_ord[k - 1]);
			float q = ((float)NAP_PRIOR_K * q_nn + suffix[k]) /
				  ((float)NAP_PRIOR_K + total);
			float debt_penalty;

			if (d->log2_tres[k] > sleep_log2)
				q = 0.0f;

			if (d->cooldown_until > now_ns &&
			    d->debt[k] > 0)
				q *= 0.5f;

			debt_penalty = (float)d->debt[k] * 0.1f;
			debt_penalty = (debt_penalty > 0.9f) ? 0.9f : debt_penalty;
			q *= (1.0f - debt_penalty);

			if (q < qmin)
				qmin = q;
			q = qmin;

			if (q >= conf)
				m = k;
			else
				break;
		}

		for (k = m; k >= 1; k--) {
			if (dev->states_usage[k].disable)
				continue;
			if ((u64)drv->states[k].exit_latency * NSEC_PER_USEC > latency_req)
				continue;
			idx = k;
			break;
		}
		return idx;
	}
}
