# CPQ IO Scheduler

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Kernel](https://img.shields.io/badge/Linux-5.15%2B-orange.svg)](https://kernel.org)

CPQ (Custom Priority Queue) is a Linux blk-mq IO scheduler that combines priority-aware dispatch with thread-group–oriented fairness and low CPU overhead, targeting mobile and embedded storage.

## Overview

CPQ originates reverse-engineering of Xiaomi's proprietary `cpq.ko` scheduler, reconstructed from a stripped ARM64 kernel module.
The implementation supports modern multi-queue kernels and is designed to balance interactive responsiveness, throughput, and fairness on modern mobile devices.

## Features

- **Priority-aware dispatch** across three IO classes (RT, BE, IDLE), mapped from Linux `ioprio` semantics
- **Foreground and background group separation**, allowing interactive workloads and batch jobs to be handled with different timeouts and expectations
- **Per-priority, per-group queues** built from three structures: a sector-ordered red–black tree, a deadline-ordered FIFO list, and a hash table for fast merge lookups
- **Deadline enforcement and priority aging** to cap request latency and prevent long-term starvation of lower-priority flows
- **Lightweight fairness** via per-thread-group accounting and throttling, accelerated by a per-CPU cache of hot group statistics to keep scheduling overhead small
- **Standard blk-mq elevator integration** with sysfs attributes for all key tunables under `/sys/block/<dev>/queue/iosched/`

## Building and Installation

The code is provided as an out-of-tree module `cpq-iosched.c` and uses standard Linux kernel headers for the block layer and scheduler APIs.  
It has been built and tested on Linux 6.1 (Ubuntu 22.04, RK3588 SoC, eMMC 5.1) and should be portable to recent 5.x/6.x kernels that use blk-mq.

### Enable CPQ on a Device

To enable CPQ on a specific device at runtime:

```
# Replace <dev> with your block device, e.g. mmcblk0 or sda
echo cpq | sudo tee /sys/block/<dev>/queue/scheduler
cat /sys/block/<dev>/queue/scheduler
```

To make CPQ the default scheduler via the kernel command line, add `elevator=cpq` to your bootloader configuration (for example through the GRUB `GRUB_CMDLINE_LINUX` setting) and regenerate the config.

## Configuration and Usage

CPQ exposes its policy knobs through standard elevator attributes under `/sys/block/<dev>/queue/iosched/`, allowing live tuning without rebuilding the kernel.  
The most important parameters are listed below; names correspond to sysfs files.

### Key Tunables

| Attribute           | Default (approx.)         | Description                                                                 |
|---------------------|---------------------------|-----------------------------------------------------------------------------|
| `read_expire`       | 500 ms                    | Deadline for read requests before they are considered urgent                |
| `write_expire`      | 5000 ms                   | Deadline for write requests, allowing more batching than reads              |
| `writes_starved`    | 2                         | Max consecutive read batches before forcing a write batch                   |
| `fifo_batch`        | 16                        | Number of requests dispatched in one batch from a queue                     |
| `async_depth`       | Queue-dependent           | Effective depth limit for asynchronous IO on each hardware queue            |
| `front_merges`      | 1 (enabled)               | Controls whether front merges are attempted in addition to back merges      |
| `prio_aging_expire` | 10 s                      | Time after which a request is "aged" to increase its effective priority     |
| `fore_timeout`      | 2 s                       | Activity timeout for the foreground group, tuned for responsiveness         |
| `back_timeout`      | 125 ms                    | Timeout governing background group accounting and idling                    |
| `slice_idle`        | 1 ms                      | Idle slice duration before the scheduler considers switching direction      |
| `io_threshold`      | 1 ms (nanosecond scale)   | Threshold used for IO timing-based heuristics inside CPQ                    |
| `cpq_log`           | 0 (off)                   | Internal logging/debug toggle for CPQ-specific diagnostics                  |

### Usage Scenarios

**Latency-sensitive workloads** (databases, UI-critical services):

- Reduce `read_expire` and `fifo_batch`
- Keep strict deadlines to minimize tail latency

**Throughput-oriented batch jobs**:

- Increase `fifo_batch` and relax `read_expire`
- Allow larger sequential runs and better device utilization

## Benchmarks and Publication

CPQ has been evaluated on an RK3588-based system with 8 GB LPDDR4x, 32 GB eMMC 5.1, Ubuntu 22.04, and Linux 6.1.43, using six synthetic and mixed workloads (standard mixed, ionice low/high, mixed async, and two application-style tests).

### Performance Results

Across these workloads CPQ wins against `mq-deadline` in **5 out of 6 cases (~83% win rate)**, with:

- **Average IOPS uplift of ~1.9%**
- **Up to ~4% improvement on mixed IO**
- **CPU overhead within ~0.4 percentage points**

## License and Acknowledgments

**GPL-2.0** | Copyright (C) 2025 Rasenkai <rasenkai99@gmail.com>

The CPQ IO scheduler is licensed under **GPL-2.0**; the source file carries the SPDX identifier `GPL-2.0` and uses the standard `MODULE_LICENSE("GPL")` tag.

This work builds on:

- Xiaomi's original CPQ binary
- The Linux blk-mq block layer
- Ideas from Samsung's SSG scheduler for thread-group–based fairness

If you use CPQ in academic work, please cite the repository. If you deploy it in production, consider sharing feedback and workloads to guide future improvements.
