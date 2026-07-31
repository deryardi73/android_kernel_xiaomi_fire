#ifndef NAP_H
#define NAP_H

#include <linux/cpuidle.h>
#include <linux/jump_label.h>
#include <linux/ktime.h>
#include <asm/neon.h>

#define NAP_INPUT_SIZE    8
#define NAP_HIDDEN_SIZE   8
#define NAP_NUM_CUTS      (CPUIDLE_STATE_MAX - 1)

struct nap_weights {
	float w_h1[NAP_INPUT_SIZE][NAP_HIDDEN_SIZE];
	float b_h1[NAP_HIDDEN_SIZE];
	float w_out[NAP_HIDDEN_SIZE];
	float b_out;
	float thr_ord[NAP_NUM_CUTS];
} __aligned(32);

#define NAP_HISTORY_SIZE     8

#define NAP_MIN_STATE_REFRESH_JIFFIES  HZ

struct nap_stats {
	u64 total_selects;
	u64 total_residency_ns;
	u64 overshoot_count;
	u64 learn_count;
};

struct nap_cpu_data {
	u64   history[NAP_HISTORY_SIZE];
	float log_history[NAP_HISTORY_SIZE];
	int   hist_idx;
	int   hist_count;

	u64     prev_idle_exit;
	s64     last_predicted_ns;
	s64     last_prediction_error;

	bool short_circuited;
	int  cached_min_state;
	s64  cached_min_state_latency;
	unsigned long cached_min_state_jiffies;

	unsigned long last_learn_jiffies;
	unsigned int  learn_jiffies_min;

	int   last_selected_idx;

	float nn_output;

	float hidden_out[NAP_HIDDEN_SIZE] __aligned(32);
	float features_f32[NAP_INPUT_SIZE] __aligned(32);

	float learn_d_out;
	float learn_lr;
	float learn_d_hid[NAP_HIDDEN_SIZE] __aligned(32);

	float log2_tres[CPUIDLE_STATE_MAX];

	float bin_count[CPUIDLE_STATE_MAX];

	bool  needs_learn;
	bool  have_sample;
	u64   learn_actual_ns;

	struct nap_weights weights;
	struct nap_weights *active_w;

	unsigned int learning_rate_millths;
	unsigned int max_grad_norm_millths;
	unsigned int conf_millths;
	int   learn_interval;
	int   learn_counter;
	bool reset_pending;

	u64   debt[CPUIDLE_STATE_MAX];
	u64   debt_decay_ns;
	u64   cooldown_until;

	struct nap_stats stats;
} __aligned(64);

static inline float fclampf(float v, float lo, float hi)
{
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

void nap_nn_forward_neon(const float *input, float *output,
			  float *hidden_save, const struct nap_weights *w);
void nap_nn_learn_neon(struct nap_cpu_data *d);

DECLARE_PER_CPU(struct nap_cpu_data, nap_data);

int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d);

int  nap_sysfs_init(void);
void nap_sysfs_exit(void);

#endif
