/*
 * Zen IO scheduler - blk-mq port with eMMC-oriented additions
 *
 * Original Zen: Copyright (C) 2012 Brandon Berhent <bbedward@gmail.com>
 * Primarily based on Noop, deadline, and SIO IO schedulers.
 *
 * ----------------------------------------------------------------------
 * This is a from-scratch port of Zen from the legacy single-queue
 * elevator_ops interface to the blk-mq elevator_mq_ops interface,
 * because this tree's eMMC block queue (drivers/mmc/core/queue.c) is
 * hardcoded to blk_mq_init_queue() -- legacy .ops.sq schedulers can
 * never attach to it regardless of how correctly they're written.
 *
 * The structure/dispatch/init/exit skeleton below mirrors
 * block/mq-deadline.c from this same tree (the only confirmed-working
 * elevator_mq_ops reference available), since mq-deadline is the
 * canonical example of how to drive rb-tree + fifo + rq-hash state
 * correctly under the blk-mq scheduling framework.
 *
 * Original Zen's core policy is preserved as-is:
 *   - two priority buckets, "SYNC" checked before "ASYNC"
 *   - soft per-bucket expiry (sync_expire / async_expire)
 *   - batching counter that, once it exceeds fifo_batch, forces a
 *     fifo-expiry check across both buckets before continuing
 *
 * NOTE ON BUCKET NAMING: exactly like the original source, "SYNC"/
 * "ASYNC" here are just labels for rq_data_dir()'s two buckets
 * (0/1), not the REQ_SYNC bio flag. rq_data_dir() in this tree
 * resolves to (op_is_write(...) ? WRITE : READ), so bucket SYNC(1)
 * is actually the WRITE bucket and ASYNC(0) is the READ bucket.
 *
 * New in this port (eMMC has no NCQ / no hardware command reordering,
 * so software-side request shaping matters far more here than it
 * would on an SSD/UFS device with deep hardware queues):
 *
 *   1. Sequential merging (rb-tree + rq-hash), via the same
 *      request_merge/bio_merge/requests_merged/request_merged
 *      machinery mq-deadline uses. The original 2012 Zen had *no*
 *      merging at all beyond whatever the generic plug layer did
 *      before requests reached the elevator -- every bio became its
 *      own request. On a device that processes commands strictly
 *      serially, turning N adjacent small requests into 1 larger one
 *      is the single biggest win available.
 *   2. Discard/TRIM isolation: REQ_OP_DISCARD requests (fstrim,
 *      filesystem discard) are routed to a separate low-priority
 *      list instead of the normal sync/async buckets, so a discard
 *      burst can't delay foreground reads. A starvation counter
 *      (DISCARD_STARVE_LIMIT) forces one discard through periodically
 *      even under sustained load, since on an actively-used phone the
 *      "only dispatch when both fifo lists are empty" condition can
 *      go unmet indefinitely and fstrim would effectively never
 *      finish.
 *   3. Read/write starvation control: the original bucket-selection
 *      logic always drained the WRITE bucket (labeled "SYNC") before
 *      the READ bucket ("ASYNC"), with no limit. On a device with a
 *      single in-flight command slot (no NCQ / no CQHCI), that means
 *      a sustained write burst (photo save, app install, journal
 *      commit) could delay every foreground read with nothing to cap
 *      it. This now mirrors mq-deadline's approach in this same
 *      tree: reads are served first, and a `writes_starved` counter
 *      (default 2, tunable) forces a write through once reads have
 *      starved it that many times in a row.
 *   4. Metadata/priority fast path: REQ_META and REQ_PRIO requests
 *      (filesystem journal commits, inode/dir metadata) are routed to
 *      their own list and dispatched ahead of the normal read/write
 *      batching entirely, the same way discards are isolated but at
 *      the opposite end of the priority order. These are usually
 *      small and rarely benefit much from sequential merging anyway,
 *      so skipping the rb-tree/hash for them trades a small amount of
 *      merge opportunity for a large latency win on fsync-heavy
 *      operations that would otherwise block behind a write batch.
 *
 * Ideas not included in this pass (flag if you want them added):
 *   - adaptive fifo_batch sizing based on recent request size/streak
 *   - a deferred-work idle-flush for pending writes
 * ----------------------------------------------------------------------
 */
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/fs.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/elevator.h>
#include <linux/bio.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/rbtree.h>

#include "blk.h"
#include "blk-mq.h"
#include "blk-mq-sched.h"

enum zen_data_dir { ASYNC, SYNC };

static const int sync_expire  = HZ / 4;    /* max time before a sync is submitted. */
static const int async_expire = 2 * HZ;    /* ditto for async, these limits are SOFT! */
static const int fifo_batch = 16;          /* sequential requests treated as one batch */
static const int writes_starved = 2;       /* max times reads can starve a write */
/*
 * 8ms: reasonable single-command-slot eMMC write completion target.
 * Above this, dynamic_batch shrinks so overdue requests get rechecked
 * sooner instead of waiting behind a full write batch.
 */
static const int target_write_lat_ns = 8 * 1000 * 1000;

/*
 * Non-tunable safety valve: once this many requests have been
 * dispatched from any source while the discard list is non-empty,
 * force one discard through regardless of what else is pending.
 * Kept as a fixed constant instead of a sysfs knob on purpose --
 * this is a correctness/fairness backstop, not a performance dial.
 */
#define DISCARD_STARVE_LIMIT 128

struct zen_data {
	/* requests are present on both sort_list and fifo_list */
	struct rb_root sort_list[2];
	struct list_head fifo_list[2];

	/* discards get their own low-priority, unmerged queue */
	struct list_head discard_list;

	/*
	 * metadata/priority requests (REQ_META | REQ_PRIO) get their
	 * own high-priority, unmerged queue -- dispatched ahead of
	 * everything except an explicit at_head insert.
	 */
	struct list_head meta_list;

	struct list_head dispatch;

	unsigned int batching;		/* number of sequential requests made */

	/* read/write starvation bookkeeping */
	unsigned int starved;		/* times reads have starved a write */

	/* discard starvation bookkeeping */
	unsigned int discard_batching;	/* dispatches since a discard last went out */

	/*
	 * v3: adaptive batch throttling. dynamic_batch is the effective
	 * threshold actually used in place of fifo_batch -- it shrinks
	 * when observed write completion latency rises above target
	 * (device is busy/slow, so check for overdue reads/writes sooner)
	 * and grows back toward fifo_batch when latency is healthy. This
	 * is the same spirit as kyber's latency-target throttling, scaled
	 * down to a single knob instead of a full per-class token bucket.
	 */
	u64 avg_write_lat_ns;		/* EWMA of write completion latency */
	unsigned int dynamic_batch;	/* current adaptive batching threshold */

	/* tunables */
	int fifo_expire[2];
	int fifo_batch;
	int front_merges;
	int writes_starved;
	int target_write_lat_ns;	/* latency target driving dynamic_batch */

	spinlock_t lock;
};

static inline struct rb_root *
zen_rb_root(struct zen_data *zdata, struct request *rq)
{
	return &zdata->sort_list[rq_data_dir(rq)];
}

static inline void
zen_add_rq_rb(struct zen_data *zdata, struct request *rq)
{
	elv_rb_add(zen_rb_root(zdata, rq), rq);
}

static inline void
zen_del_rq_rb(struct zen_data *zdata, struct request *rq)
{
	elv_rb_del(zen_rb_root(zdata, rq), rq);
}

/*
 * rq_fifo_time()/rq_set_fifo_time() aren't provided globally by
 * elevator.h in this tree -- see mq-deadline.c/deadline-iosched.c,
 * every in-tree scheduler defines its own accessor for the u64
 * rq->fifo_time field.
 */
static inline void rq_set_fifo_time(struct request *rq, unsigned long expire)
{
	rq->fifo_time = expire;
}

static inline unsigned long rq_fifo_time(struct request *rq)
{
	return (unsigned long)rq->fifo_time;
}

/*
 * remove rq from rbtree, fifo and hash.
 */
static void zen_remove_request(struct request_queue *q, struct request *rq)
{
	struct zen_data *zdata = q->elevator->elevator_data;

	list_del_init(&rq->queuelist);

	/* we might not be on the rbtree, if we are doing an insert merge */
	if (!RB_EMPTY_NODE(&rq->rb_node))
		zen_del_rq_rb(zdata, rq);

	elv_rqhash_del(q, rq);
	if (q->last_merge == rq)
		q->last_merge = NULL;
}

static void zen_request_merged(struct request_queue *q, struct request *req,
				enum elv_merge type)
{
	struct zen_data *zdata = q->elevator->elevator_data;

	/* if the merge was a front merge, we need to reposition request */
	if (type == ELEVATOR_FRONT_MERGE) {
		zen_del_rq_rb(zdata, req);
		zen_add_rq_rb(zdata, req);
	}
}

static void
zen_merged_requests(struct request_queue *q, struct request *rq,
		    struct request *next)
{
	/*
	 * if next expires before rq, assign its expire time to rq
	 * and move into next position (next will be deleted) in fifo
	 */
	if (!list_empty(&rq->queuelist) && !list_empty(&next->queuelist)) {
		if (time_before(rq_fifo_time(next), rq_fifo_time(rq))) {
			list_move(&rq->queuelist, &next->queuelist);
			rq_set_fifo_time(rq, rq_fifo_time(next));
		}
	}

	/* next request is gone -- clean it out of rbtree/hash too */
	zen_remove_request(q, next);
}

static int zen_request_merge(struct request_queue *q, struct request **rq,
			     struct bio *bio)
{
	struct zen_data *zdata = q->elevator->elevator_data;
	sector_t sector = bio_end_sector(bio);
	struct request *__rq;

	if (!zdata->front_merges)
		return ELEVATOR_NO_MERGE;

	__rq = elv_rb_find(&zdata->sort_list[bio_data_dir(bio)], sector);
	if (__rq && elv_bio_merge_ok(__rq, bio)) {
		*rq = __rq;
		return ELEVATOR_FRONT_MERGE;
	}

	return ELEVATOR_NO_MERGE;
}

static bool zen_bio_merge(struct blk_mq_hw_ctx *hctx, struct bio *bio)
{
	struct request_queue *q = hctx->queue;
	struct zen_data *zdata = q->elevator->elevator_data;
	struct request *free = NULL;
	bool ret;

	spin_lock(&zdata->lock);
	ret = blk_mq_sched_try_merge(q, bio, &free);
	spin_unlock(&zdata->lock);

	if (free)
		blk_mq_free_request(free);

	return ret;
}

/*
 * add rq to rbtree, hash and fifo. Metadata/priority requests bypass
 * merge bookkeeping and go straight to their own high-priority list;
 * discards go to their own low-priority list. Both skip merge
 * bookkeeping entirely -- there's little to gain merging trim ranges
 * or small metadata blocks here, and keeping them off the rb-tree/
 * hash keeps them from ever being picked as a front-merge target for
 * real read/write bios.
 */
static void zen_insert_request(struct blk_mq_hw_ctx *hctx, struct request *rq,
			       bool at_head)
{
	struct request_queue *q = hctx->queue;
	struct zen_data *zdata = q->elevator->elevator_data;
	const int dir = rq_data_dir(rq);

	if (rq->cmd_flags & (REQ_META | REQ_PRIO)) {
		list_add_tail(&rq->queuelist, &zdata->meta_list);
		return;
	}

	if (op_is_discard(req_op(rq))) {
		list_add_tail(&rq->queuelist, &zdata->discard_list);
		return;
	}

	if (blk_mq_sched_try_insert_merge(q, rq))
		return;

	blk_mq_sched_request_inserted(rq);

	if (at_head || blk_rq_is_passthrough(rq)) {
		if (at_head)
			list_add(&rq->queuelist, &zdata->dispatch);
		else
			list_add_tail(&rq->queuelist, &zdata->dispatch);
		return;
	}

	zen_add_rq_rb(zdata, rq);

	if (rq_mergeable(rq)) {
		elv_rqhash_add(q, rq);
		if (!q->last_merge)
			q->last_merge = rq;
	}

	rq_set_fifo_time(rq, jiffies + zdata->fifo_expire[dir]);
	list_add_tail(&rq->queuelist, &zdata->fifo_list[dir]);
}

static void zen_insert_requests(struct blk_mq_hw_ctx *hctx,
				struct list_head *list, bool at_head)
{
	struct request_queue *q = hctx->queue;
	struct zen_data *zdata = q->elevator->elevator_data;

	spin_lock(&zdata->lock);
	while (!list_empty(list)) {
		struct request *rq;

		rq = list_first_entry(list, struct request, queuelist);
		list_del_init(&rq->queuelist);
		zen_insert_request(hctx, rq, at_head);
	}
	spin_unlock(&zdata->lock);
}

/*
 * get the first expired request in direction ddir
 */
static struct request *
zen_expired_request(struct zen_data *zdata, int ddir)
{
	struct request *rq;

	if (list_empty(&zdata->fifo_list[ddir]))
		return NULL;

	rq = rq_entry_fifo(zdata->fifo_list[ddir].next);
	if (time_after(jiffies, rq_fifo_time(rq)))
		return rq;

	return NULL;
}

/*
 * zen_check_fifo returns whichever of the two buckets' head request
 * is more overdue, or NULL if neither has expired.
 */
static struct request *
zen_check_fifo(struct zen_data *zdata)
{
	struct request *rq_sync = zen_expired_request(zdata, SYNC);
	struct request *rq_async = zen_expired_request(zdata, ASYNC);

	if (rq_async && rq_sync) {
		if (time_after(rq_fifo_time(rq_async), rq_fifo_time(rq_sync)))
			return rq_sync;
		return rq_async;
	} else if (rq_sync) {
		return rq_sync;
	} else if (rq_async) {
		return rq_async;
	}

	return NULL;
}

/*
 * Choose the next request to dispatch out of the two direction
 * buckets. ASYNC(0) is the READ bucket and SYNC(1) is the WRITE
 * bucket (see the bucket-naming note in the file header). Reads are
 * served first since this device has a single in-flight command slot
 * and foreground reads are what the user is waiting on -- but writes
 * are guaranteed a turn every `writes_starved` reads so a sustained
 * write burst can't lock them out indefinitely. This mirrors
 * mq-deadline's dd_dispatch_request() in this same tree.
 */
static struct request *
zen_choose_request(struct zen_data *zdata)
{
	bool reads  = !list_empty(&zdata->fifo_list[ASYNC]);
	bool writes = !list_empty(&zdata->fifo_list[SYNC]);

	if (reads) {
		if (writes && zdata->starved++ >= zdata->writes_starved)
			goto dispatch_writes;

		return rq_entry_fifo(zdata->fifo_list[ASYNC].next);
	}

	if (writes) {
dispatch_writes:
		zdata->starved = 0;
		return rq_entry_fifo(zdata->fifo_list[SYNC].next);
	}

	return NULL;
}

static void zen_move_request(struct zen_data *zdata, struct request *rq)
{
	zen_remove_request(rq->q, rq);
}

static struct request *__zen_dispatch_request(struct zen_data *zdata)
{
	struct request *rq = NULL;

	if (!list_empty(&zdata->dispatch)) {
		rq = list_first_entry(&zdata->dispatch, struct request,
				       queuelist);
		list_del_init(&rq->queuelist);
		goto done;
	}

	/* metadata/priority requests preempt everything else */
	if (!list_empty(&zdata->meta_list)) {
		rq = list_first_entry(&zdata->meta_list, struct request,
				       queuelist);
		list_del_init(&rq->queuelist);
		goto done;
	}

	/*
	 * discard starvation backstop: if discards have been waiting
	 * behind normal traffic for too long, force one through now
	 * instead of waiting for both fifo lists to fully drain.
	 */
	if (!list_empty(&zdata->discard_list) &&
	    zdata->discard_batching++ >= DISCARD_STARVE_LIMIT) {
		zdata->discard_batching = 0;
		rq = list_first_entry(&zdata->discard_list,
				struct request, queuelist);
		list_del_init(&rq->queuelist);
		rq->rq_flags |= RQF_STARTED;
		return rq;
	}

	/* check for and issue expired requests first */
	if (zdata->batching > zdata->dynamic_batch) {
		zdata->batching = 0;
		rq = zen_check_fifo(zdata);
	}

	if (!rq) {
		rq = zen_choose_request(zdata);
		if (!rq) {
			/* nothing but discards left -- drain those */
			if (!list_empty(&zdata->discard_list)) {
				zdata->discard_batching = 0;
				rq = list_first_entry(&zdata->discard_list,
						struct request, queuelist);
				list_del_init(&rq->queuelist);
				rq->rq_flags |= RQF_STARTED;
				return rq;
			}
			return NULL;
		}
	}

	zdata->batching++;
	zen_move_request(zdata, rq);
done:
	rq->rq_flags |= RQF_STARTED;
	return rq;
}

/*
 * v3 adaptive hook: called by blk-mq when a request completes.
 * Tracks write completion latency (io_start_time_ns -> now) as an
 * EWMA and steers dynamic_batch toward target_write_lat_ns. Reads are
 * ignored here -- what we're protecting against is a slow write
 * burst delaying everything behind it, so write latency is the
 * signal that matters on a single-command-slot device.
 */
static void zen_completed_request(struct request *rq)
{
	struct zen_data *zdata = rq->q->elevator->elevator_data;
	u64 lat;

	if (rq_data_dir(rq) != WRITE || !rq->io_start_time_ns)
		return;

	lat = ktime_get_ns() - rq->io_start_time_ns;

	/* EWMA, 1/8 weight on the new sample -- same smoothing constant
	 * the classic Linux loadavg uses, cheap and stable enough here.
	 */
	if (!zdata->avg_write_lat_ns)
		zdata->avg_write_lat_ns = lat;
	else
		zdata->avg_write_lat_ns =
			(zdata->avg_write_lat_ns * 7 + lat) / 8;

	if (zdata->avg_write_lat_ns > zdata->target_write_lat_ns) {
		if (zdata->dynamic_batch > 1)
			zdata->dynamic_batch--;
	} else if (zdata->dynamic_batch < zdata->fifo_batch) {
		zdata->dynamic_batch++;
	}
}

static struct request *zen_dispatch_request(struct blk_mq_hw_ctx *hctx)
{
	struct zen_data *zdata = hctx->queue->elevator->elevator_data;
	struct request *rq;

	spin_lock(&zdata->lock);
	rq = __zen_dispatch_request(zdata);
	spin_unlock(&zdata->lock);

	return rq;
}

static bool zen_has_work(struct blk_mq_hw_ctx *hctx)
{
	struct zen_data *zdata = hctx->queue->elevator->elevator_data;

	return !list_empty_careful(&zdata->dispatch) ||
		!list_empty_careful(&zdata->fifo_list[SYNC]) ||
		!list_empty_careful(&zdata->fifo_list[ASYNC]) ||
		!list_empty_careful(&zdata->meta_list) ||
		!list_empty_careful(&zdata->discard_list);
}

static void zen_prepare_request(struct request *rq, struct bio *bio)
{
	/* nothing to do; only needed so .finish_request gets called */
}

static void zen_finish_request(struct request *rq)
{
}

static void zen_exit_queue(struct elevator_queue *e)
{
	struct zen_data *zdata = e->elevator_data;

	BUG_ON(!list_empty(&zdata->fifo_list[SYNC]));
	BUG_ON(!list_empty(&zdata->fifo_list[ASYNC]));
	kfree(zdata);
}

static int zen_init_queue(struct request_queue *q, struct elevator_type *e)
{
	struct zen_data *zdata;
	struct elevator_queue *eq;

	eq = elevator_alloc(q, e);
	if (!eq)
		return -ENOMEM;

	zdata = kzalloc_node(sizeof(*zdata), GFP_KERNEL, q->node);
	if (!zdata) {
		kobject_put(&eq->kobj);
		return -ENOMEM;
	}
	eq->elevator_data = zdata;

	INIT_LIST_HEAD(&zdata->fifo_list[SYNC]);
	INIT_LIST_HEAD(&zdata->fifo_list[ASYNC]);
	INIT_LIST_HEAD(&zdata->discard_list);
	INIT_LIST_HEAD(&zdata->meta_list);
	INIT_LIST_HEAD(&zdata->dispatch);
	zdata->sort_list[SYNC] = RB_ROOT;
	zdata->sort_list[ASYNC] = RB_ROOT;
	zdata->fifo_expire[SYNC] = sync_expire;
	zdata->fifo_expire[ASYNC] = async_expire;
	zdata->fifo_batch = fifo_batch;
	zdata->front_merges = 1;
	zdata->writes_starved = writes_starved;
	zdata->target_write_lat_ns = target_write_lat_ns;
	zdata->dynamic_batch = fifo_batch;	/* start optimistic, full batch */
	zdata->avg_write_lat_ns = 0;
	zdata->starved = 0;
	zdata->discard_batching = 0;
	spin_lock_init(&zdata->lock);

	q->elevator = eq;
	return 0;
}

/* Sysfs */
static ssize_t
zen_var_show(int var, char *page)
{
	return sprintf(page, "%d\n", var);
}

static void
zen_var_store(int *var, const char *page)
{
	char *p = (char *)page;

	*var = simple_strtol(p, &p, 10);
}

#define SHOW_FUNCTION(__FUNC, __VAR, __CONV) \
static ssize_t __FUNC(struct elevator_queue *e, char *page) \
{ \
	struct zen_data *zdata = e->elevator_data; \
	int __data = __VAR; \
	if (__CONV) \
		__data = jiffies_to_msecs(__data); \
	return zen_var_show(__data, (page)); \
}
SHOW_FUNCTION(zen_sync_expire_show, zdata->fifo_expire[SYNC], 1);
SHOW_FUNCTION(zen_async_expire_show, zdata->fifo_expire[ASYNC], 1);
SHOW_FUNCTION(zen_fifo_batch_show, zdata->fifo_batch, 0);
SHOW_FUNCTION(zen_front_merges_show, zdata->front_merges, 0);
SHOW_FUNCTION(zen_writes_starved_show, zdata->writes_starved, 0);
SHOW_FUNCTION(zen_target_write_lat_ns_show, zdata->target_write_lat_ns, 0);
#undef SHOW_FUNCTION

#define STORE_FUNCTION(__FUNC, __PTR, MIN, MAX, __CONV) \
static ssize_t __FUNC(struct elevator_queue *e, const char *page, size_t count) \
{ \
	struct zen_data *zdata = e->elevator_data; \
	int __data; \
	zen_var_store(&__data, (page)); \
	if (__data < (MIN)) \
		__data = (MIN); \
	else if (__data > (MAX)) \
		__data = (MAX); \
	if (__CONV) \
		*(__PTR) = msecs_to_jiffies(__data); \
	else \
		*(__PTR) = __data; \
	return count; \
}
STORE_FUNCTION(zen_sync_expire_store, &zdata->fifo_expire[SYNC], 0, INT_MAX, 1);
STORE_FUNCTION(zen_async_expire_store, &zdata->fifo_expire[ASYNC], 0, INT_MAX, 1);
STORE_FUNCTION(zen_fifo_batch_store, &zdata->fifo_batch, 0, INT_MAX, 0);
STORE_FUNCTION(zen_front_merges_store, &zdata->front_merges, 0, 1, 0);
STORE_FUNCTION(zen_writes_starved_store, &zdata->writes_starved, INT_MIN, INT_MAX, 0);
STORE_FUNCTION(zen_target_write_lat_ns_store, &zdata->target_write_lat_ns, 1, INT_MAX, 0);
#undef STORE_FUNCTION

#define DD_ATTR(name) \
	__ATTR(name, 0644, zen_##name##_show, zen_##name##_store)

static struct elv_fs_entry zen_attrs[] = {
	DD_ATTR(sync_expire),
	DD_ATTR(async_expire),
	DD_ATTR(fifo_batch),
	DD_ATTR(front_merges),
	DD_ATTR(writes_starved),
	DD_ATTR(target_write_lat_ns),
	__ATTR_NULL
};

static struct elevator_type iosched_zen = {
	.ops.mq = {
		.insert_requests	= zen_insert_requests,
		.dispatch_request	= zen_dispatch_request,
		.completed_request	= zen_completed_request,
		.prepare_request	= zen_prepare_request,
		.finish_request		= zen_finish_request,
		.next_request		= elv_rb_latter_request,
		.former_request		= elv_rb_former_request,
		.bio_merge		= zen_bio_merge,
		.request_merge		= zen_request_merge,
		.requests_merged	= zen_merged_requests,
		.request_merged		= zen_request_merged,
		.has_work		= zen_has_work,
		.init_sched		= zen_init_queue,
		.exit_sched		= zen_exit_queue,
	},

	.uses_mq	= true,
	.elevator_attrs = zen_attrs,
	.elevator_name = "mq-zen",
	.elevator_owner = THIS_MODULE,
};

static int __init zen_init(void)
{
	return elv_register(&iosched_zen);
}

static void __exit zen_exit(void)
{
	elv_unregister(&iosched_zen);
}

module_init(zen_init);
module_exit(zen_exit);

MODULE_AUTHOR("Brandon Berhent, blk-mq port & eMMC additions");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Zen IO scheduler - blk-mq port with eMMC + adaptive batching");
MODULE_VERSION("3.0-mq");