/* SPDX-License-Identifier: GPL-2.0 */
/*
 * arm64 KFENCE support. Backported to 4.19 for inferno kernel (fire/MT6768).
 *
 * Reuses set_memory_valid() from arch/arm64/mm/pageattr.c, already present
 * on this tree (used by CONFIG_DEBUG_PAGEALLOC), so no arch-side plumbing
 * needed beyond this header.
 *
 * Copyright (C) 2020, Google LLC.
 */

#ifndef __ASM_KFENCE_H
#define __ASM_KFENCE_H

#include <asm/cacheflush.h> /* for set_memory_valid() */

static inline bool arch_kfence_init_pool(void) { return true; }

static inline bool kfence_protect_page(unsigned long addr, bool protect)
{
	set_memory_valid(addr, 1, !protect);

	return true;
}

#endif /* __ASM_KFENCE_H */