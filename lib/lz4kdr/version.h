/* SPDX-License-Identifier: Dual BSD/GPL */
#ifndef _LIB_LZ4KDR_VERSION_H
#define _LIB_LZ4KDR_VERSION_H

/*
 * LZ4KDR -- version identifiers.
 *
 * Kept in lib/lz4kdr/ rather than include/linux/lz4kdr.h so that bumping
 * LZ4KDR_VERSION (the only string that changes from one release to the
 * next) does not invalidate the ccache entry for every translation unit
 * that includes <linux/lz4kdr.h> (crypto/lz4kdr.c,
 * drivers/block/zram/backend_lz4kdr.c, etc.). Only lz4kdr_encode.c
 * includes this header, so a version bump rebuilds just that one file.
 */

#define LZ4KDR_PROGNAME	"LZ4KDR (speed-tuned LZ4KD derivative)"
#define LZ4KDR_AUTHOR	"Masahito Suzuki"

#define LZ4KDR_VERSION	"1.3"

#endif /* _LIB_LZ4KDR_VERSION_H */
