/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Kernel Electric-Fence (KFENCE). Internal header. Backported to 4.19 for
 * inferno kernel (fire/MT6768). Based on upstream v5.12 mm/kfence/kfence.h.
 *
 * Copyright (C) 2020, Google LLC.
 */

#ifndef MM_KFENCE_KFENCE_H
#define MM_KFENCE_KFENCE_H

#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/types.h>

#include "../slab.h" /* for struct kmem_cache */

/*
 * Get the canary byte pattern for @addr. Use a pattern that varies based on
 * the lower 3 bits of the address, to detect memory corruptions with higher
 * probability, where similar constants are used.
 */
#define KFENCE_CANARY_PATTERN(addr) ((u8)0xaa ^ (u8)((unsigned long)(addr) & 0x7))

/* Maximum stack depth for reports. */
#define KFENCE_STACK_DEPTH 64

/* KFENCE object states. */
enum kfence_object_state {
	KFENCE_OBJECT_UNUSED,		/* Object is unused. */
	KFENCE_OBJECT_ALLOCATED,	/* Object is currently allocated. */
	KFENCE_OBJECT_FREED,		/* Object was allocated, and then freed. */
};

/*
 * Alloc/free tracking information. NOTE: on 4.19 we do NOT use stack_depot
 * (API mismatch: this tree's lib/stackdepot.c predates the entries-array
 * stack_depot_save()/fetch() API). Upstream KFENCE itself never used
 * stack_depot for this either -- entries are stored inline per-object via
 * stack_trace_save(), which this tree already has in kernel/stacktrace.c.
 */
struct kfence_track {
	pid_t pid;
	int num_stack_entries;
	unsigned long stack_entries[KFENCE_STACK_DEPTH];
};

/* KFENCE metadata per guarded allocation. */
struct kfence_metadata {
	struct list_head list;		/* Freelist node; access under kfence_freelist_lock. */
	struct rcu_head rcu_head;	/* For delayed freeing. */

	/*
	 * Lock protecting below data; to ensure consistency of the below data,
	 * since the following may execute concurrently: __kfence_alloc(),
	 * __kfence_free(), kfence_handle_page_fault(). However, note that we
	 * cannot grab the same metadata off the freelist twice, and multiple
	 * __kfence_alloc() cannot run concurrently on the same metadata.
	 */
	raw_spinlock_t lock;

	/* The current state of the object; see above. */
	enum kfence_object_state state;

	/*
	 * Allocated object address; cannot be calculated from size, because of
	 * alignment requirements.
	 *
	 * Invariant: ALIGN_DOWN(addr, PAGE_SIZE) is constant.
	 */
	unsigned long addr;

	/* The size of the original allocation. */
	size_t size;

	/*
	 * The kmem_cache cache of the last allocation; NULL if never allocated
	 * or the cache has already been destroyed.
	 */
	struct kmem_cache *cache;

	/*
	 * In case of an invalid access, the page that was unprotected; we
	 * optimistically only store one address.
	 */
	unsigned long unprotected_page;

	/* Allocation and free stack information. */
	struct kfence_track alloc_track;
	struct kfence_track free_track;
};

extern struct kfence_metadata kfence_metadata[CONFIG_KFENCE_NUM_OBJECTS];

/* KFENCE error types for report generation. */
enum kfence_error_type {
	KFENCE_ERROR_OOB,		/* Detected a out-of-bounds access. */
	KFENCE_ERROR_UAF,		/* Detected a use-after-free access. */
	KFENCE_ERROR_CORRUPTION,	/* Detected a memory corruption on free. */
	KFENCE_ERROR_INVALID,		/* Invalid access of unknown type. */
	KFENCE_ERROR_INVALID_FREE,	/* Invalid free. */
};

void kfence_report_error(unsigned long address, bool is_write, struct pt_regs *regs,
			  const struct kfence_metadata *meta, enum kfence_error_type type);

void kfence_print_object(struct seq_file *seq, const struct kfence_metadata *meta);

#endif /* MM_KFENCE_KFENCE_H */