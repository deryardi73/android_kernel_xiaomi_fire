#include "nap.h"

typedef float v4sf __attribute__((__vector_size__(16)));

static inline v4sf v4sf_load(const float *p)
{
	v4sf r;
	__builtin_memcpy(&r, p, sizeof(r));
	return r;
}

static inline void v4sf_store(float *p, v4sf v)
{
	__builtin_memcpy(p, &v, sizeof(v));
}

static inline v4sf v4sf_fmadd(v4sf a, v4sf b, v4sf c)
{
	asm("fmla %0.4s, %1.4s, %2.4s" : "+w"(c) : "w"(a), "w"(b));
	return c;
}

static inline v4sf v4sf_max(v4sf a, v4sf b)
{
	asm("fmax %0.4s, %1.4s, %2.4s" : "=w"(a) : "w"(a), "w"(b));
	return a;
}

static inline v4sf v4sf_min(v4sf a, v4sf b)
{
	asm("fmin %0.4s, %1.4s, %2.4s" : "=w"(a) : "w"(a), "w"(b));
	return a;
}

static inline v4sf v4sf_andnot(v4sf mask, v4sf val)
{
	asm("bic %0.16b, %1.16b, %2.16b" : "=w"(val) : "w"(val), "w"(mask));
	return val;
}

static inline v4sf v4sf_sel(v4sf mask, v4sf if_true, v4sf if_false)
{
	asm("bsl %0.16b, %1.16b, %2.16b"
	    : "=w"(mask) : "w"(if_true), "w"(if_false));
	return mask;
}

void nap_nn_forward_neon(const float *input,
			  float *output,
			  float *hidden_save,
			  const struct nap_weights *w)
{
	v4sf zero = (v4sf){ 0.0f, 0.0f, 0.0f, 0.0f };
	v4sf acc0, acc1;
	int j;

	acc0 = v4sf_load(&w->b_h1[0]);
	acc1 = v4sf_load(&w->b_h1[4]);

	for (j = 0; j < NAP_INPUT_SIZE; j++) {
		v4sf x = (v4sf){ input[j], input[j], input[j], input[j] };
		acc0 = v4sf_fmadd(v4sf_load(&w->w_h1[j][0]), x, acc0);
		acc1 = v4sf_fmadd(v4sf_load(&w->w_h1[j][4]), x, acc1);
	}

	{
		v4sf h0 = v4sf_max(acc0, zero);
		v4sf h1 = v4sf_max(acc1, zero);

		v4sf_store(&hidden_save[0], h0);
		v4sf_store(&hidden_save[4], h1);

		{
			v4sf p0 = v4sf_load(&w->w_out[0]) * h0;
			v4sf p1 = v4sf_load(&w->w_out[4]) * h1;
			v4sf s4 = p0 + p1;

			*output = s4[0] + s4[1] + s4[2] + s4[3] + w->b_out;
		}
	}
}

void nap_nn_learn_neon(struct nap_cpu_data *d)
{
	float d_out_scalar = d->learn_d_out;
	float *d_hid = d->learn_d_hid;
	float lr = d->learn_lr;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v4sf v_neg_lr = (v4sf){ -lr, -lr, -lr, -lr };
	v4sf v_cl_hi  = (v4sf){ clamp_val, clamp_val, clamp_val, clamp_val };
	v4sf v_cl_lo  = (v4sf){ -clamp_val, -clamp_val, -clamp_val, -clamp_val };
	v4sf v_d_out = (v4sf){ d_out_scalar, d_out_scalar,
			       d_out_scalar, d_out_scalar };
	v4sf zero = (v4sf){ 0.0f, 0.0f, 0.0f, 0.0f };
	v4sf h0, h1, w0, w1, d_h0, d_h1, nw0, nw1;
	v4sf grad0, grad1, cw0, cw1, db0, db1;
	int i;

	h0 = v4sf_load(&d->hidden_out[0]);
	h1 = v4sf_load(&d->hidden_out[4]);
	w0 = v4sf_load(&d->active_w->w_out[0]);
	w1 = v4sf_load(&d->active_w->w_out[4]);

	{
		v4sf cmp0, cmp1;
		v4sf grad = w0 * v_d_out;
		asm("fcmlt %0.4s, %1.4s, #0.0" : "=w"(cmp0) : "w"(h0));
		asm("fcmlt %0.4s, %1.4s, #0.0" : "=w"(cmp1) : "w"(h1));
		d_h0 = v4sf_sel(cmp0, zero, grad);
		d_h1 = v4sf_sel(cmp1, zero, w1 * v_d_out);
	}

	v4sf_store(&d_hid[0], d_h0);
	v4sf_store(&d_hid[4], d_h1);

	nw0 = v4sf_max(v4sf_min(w0 + v_neg_lr * (v_d_out * h0), v_cl_hi), v_cl_lo);
	nw1 = v4sf_max(v4sf_min(w1 + v_neg_lr * (v_d_out * h1), v_cl_hi), v_cl_lo);
	v4sf_store(&d->active_w->w_out[0], nw0);
	v4sf_store(&d->active_w->w_out[4], nw1);

	d->active_w->b_out = fclampf(
		d->active_w->b_out + fclampf(-lr * d_out_scalar,
					     -clamp_val, clamp_val),
		-clamp_val, clamp_val);

	for (i = 0; i < NAP_INPUT_SIZE; i++) {
		v4sf x = (v4sf){ d->features_f32[i], d->features_f32[i],
				 d->features_f32[i], d->features_f32[i] };
		grad0 = x * d_h0;
		grad1 = x * d_h1;
		cw0 = v4sf_load(&d->active_w->w_h1[i][0]);
		cw1 = v4sf_load(&d->active_w->w_h1[i][4]);

		v4sf_store(&d->active_w->w_h1[i][0],
			   v4sf_min(v4sf_max(cw0 + v_neg_lr * grad0,
					     v_cl_lo), v_cl_hi));
		v4sf_store(&d->active_w->w_h1[i][4],
			   v4sf_min(v4sf_max(cw1 + v_neg_lr * grad1,
					     v_cl_lo), v_cl_hi));
	}

	db0 = v4sf_min(v4sf_max(zero + v_neg_lr * d_h0, v_cl_lo), v_cl_hi);
	db1 = v4sf_min(v4sf_max(zero + v_neg_lr * d_h1, v_cl_lo), v_cl_hi);
	v4sf_store(&d->active_w->b_h1[0],
		   v4sf_load(&d->active_w->b_h1[0]) + db0);
	v4sf_store(&d->active_w->b_h1[4],
		   v4sf_load(&d->active_w->b_h1[4]) + db1);
}
