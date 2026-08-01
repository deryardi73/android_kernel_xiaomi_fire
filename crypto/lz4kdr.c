/*
 * Cryptographic API.
 *
 * LZ4KDR -- speed-tuned derivative of Huawei's LZ4KD, wired into the
 * generic crypto compress API the same way lz4/lz4hc/zstd are, so that
 * zram's zcomp_create("lz4kdr") picks it up with zero changes to the
 * zram driver itself.
 *
 * Backported from firelzrd's linux6.12.74-lz4kdr-1.3 patch, which
 * targets the newer per-algorithm "backend_NNN.c" zram driver layout
 * that doesn't exist on this 4.19 tree. lib/lz4kdr/{lz4kdr_encode.c,
 * lz4kdr_decode.c,*.h} and include/linux/lz4kdr.h are carried over
 * unmodified (they're kernel-version-agnostic); this file replaces
 * drivers/block/zram/backend_lz4kdr.{c,h} with an equivalent adapter
 * against crypto_alg/scomp_alg instead of struct zcomp_ops, following
 * this tree's crypto/lz4hc.c as the template.
 *
 * IMPORTANT (unlike lz4hc's alloc_ctx): lz4kdr_encode()'s `state`
 * scratch buffer must be zeroed once, at allocation time, and never
 * re-zeroed on later calls -- see include/linux/lz4kdr.h and
 * lz4kdr_encode.c's "change 1" comment. lz4hc/lz4's vmalloc() (no
 * zeroing) is therefore NOT safe to copy here; kzalloc() is used
 * instead, both for the required zeroing and because
 * lz4kdr_encode_state_bytes_min() is small (2KB at this tree's
 * HT_LOG2=10), well within kmalloc's normal range -- same reasoning
 * backend_lz4kdr.c gave upstream for not vmalloc()'ing it either.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/crypto.h>
#include <linux/slab.h>
#include <linux/lz4kdr.h>
#include <crypto/internal/scompress.h>

struct lz4kdr_ctx {
	void *lz4kdr_state;
};

static void *lz4kdr_alloc_ctx(struct crypto_scomp *tfm)
{
	void *ctx;

	/* kzalloc, not kmalloc/vmalloc: lz4kdr_encode() requires this
	 * buffer to be zeroed exactly once, at allocation -- see the
	 * top-of-file comment. */
	ctx = kzalloc(lz4kdr_encode_state_bytes_min(), GFP_KERNEL);
	if (!ctx)
		return ERR_PTR(-ENOMEM);

	return ctx;
}

static int lz4kdr_init(struct crypto_tfm *tfm)
{
	struct lz4kdr_ctx *ctx = crypto_tfm_ctx(tfm);

	ctx->lz4kdr_state = lz4kdr_alloc_ctx(NULL);
	if (IS_ERR(ctx->lz4kdr_state))
		return -ENOMEM;

	return 0;
}

static void lz4kdr_free_ctx(struct crypto_scomp *tfm, void *ctx)
{
	kfree(ctx);
}

static void lz4kdr_exit(struct crypto_tfm *tfm)
{
	struct lz4kdr_ctx *ctx = crypto_tfm_ctx(tfm);

	lz4kdr_free_ctx(NULL, ctx->lz4kdr_state);
}

static int __lz4kdr_compress_crypto(const u8 *src, unsigned int slen,
				     u8 *dst, unsigned int *dlen, void *ctx)
{
	int out_len = lz4kdr_encode(ctx, src, dst, slen, *dlen, 0);

	if (out_len < 0)
		return -EINVAL;
	if (out_len == 0) {
		/*
		 * INCOMPRESSIBLE (not an error): mirrors upstream's
		 * backend_lz4kdr.c raw-store signal, translated to this
		 * API's "report dlen == slen" convention -- zram's
		 * __zram_bvec_write() already special-cases comp_len ==
		 * PAGE_SIZE as a raw store (see zram_drv.c), so this
		 * takes that same path instead of failing the write.
		 */
		*dlen = slen;
		return 0;
	}

	*dlen = out_len;
	return 0;
}

static int lz4kdr_scompress(struct crypto_scomp *tfm, const u8 *src,
			     unsigned int slen, u8 *dst, unsigned int *dlen,
			     void *ctx)
{
	return __lz4kdr_compress_crypto(src, slen, dst, dlen, ctx);
}

static int lz4kdr_compress_crypto(struct crypto_tfm *tfm, const u8 *src,
				   unsigned int slen, u8 *dst,
				   unsigned int *dlen)
{
	struct lz4kdr_ctx *ctx = crypto_tfm_ctx(tfm);

	return __lz4kdr_compress_crypto(src, slen, dst, dlen,
					 ctx->lz4kdr_state);
}

static int __lz4kdr_decompress_crypto(const u8 *src, unsigned int slen,
				       u8 *dst, unsigned int *dlen, void *ctx)
{
	int out_len = lz4kdr_decode(src, dst, slen, *dlen);

	if (out_len <= 0)
		return -EINVAL;

	*dlen = out_len;
	return 0;
}

static int lz4kdr_sdecompress(struct crypto_scomp *tfm, const u8 *src,
			       unsigned int slen, u8 *dst, unsigned int *dlen,
			       void *ctx)
{
	return __lz4kdr_decompress_crypto(src, slen, dst, dlen, NULL);
}

static int lz4kdr_decompress_crypto(struct crypto_tfm *tfm, const u8 *src,
				     unsigned int slen, u8 *dst,
				     unsigned int *dlen)
{
	return __lz4kdr_decompress_crypto(src, slen, dst, dlen, NULL);
}

static struct crypto_alg alg_lz4kdr = {
	.cra_name		= "lz4kdr",
	.cra_flags		= CRYPTO_ALG_TYPE_COMPRESS,
	.cra_ctxsize		= sizeof(struct lz4kdr_ctx),
	.cra_module		= THIS_MODULE,
	.cra_list		= LIST_HEAD_INIT(alg_lz4kdr.cra_list),
	.cra_init		= lz4kdr_init,
	.cra_exit		= lz4kdr_exit,
	.cra_u			= { .compress = {
	.coa_compress		= lz4kdr_compress_crypto,
	.coa_decompress		= lz4kdr_decompress_crypto } }
};

static struct scomp_alg scomp = {
	.alloc_ctx		= lz4kdr_alloc_ctx,
	.free_ctx		= lz4kdr_free_ctx,
	.compress		= lz4kdr_scompress,
	.decompress		= lz4kdr_sdecompress,
	.base			= {
		.cra_name	= "lz4kdr",
		.cra_driver_name = "lz4kdr-scomp",
		.cra_module	 = THIS_MODULE,
	}
};

static int __init lz4kdr_mod_init(void)
{
	int ret;

	ret = crypto_register_alg(&alg_lz4kdr);
	if (ret)
		return ret;

	ret = crypto_register_scomp(&scomp);
	if (ret) {
		crypto_unregister_alg(&alg_lz4kdr);
		return ret;
	}

	return ret;
}

static void __exit lz4kdr_mod_fini(void)
{
	crypto_unregister_alg(&alg_lz4kdr);
	crypto_unregister_scomp(&scomp);
}

module_init(lz4kdr_mod_init);
module_exit(lz4kdr_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("LZ4KDR Compression Algorithm (speed-tuned LZ4KD derivative)");
MODULE_ALIAS_CRYPTO("lz4kdr");
