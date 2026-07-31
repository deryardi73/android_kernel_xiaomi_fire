# SPDX-License-Identifier: GPL-2.0-only
#

obj-$(CONFIG_CPU_IDLE_GOV_NAP) += cpuidle_gov_nap.o

cpuidle_gov_nap-y := nap.o nap_fpu.o nap_nn_neon.o

CFLAGS_REMOVE_nap_fpu.o     += -mgeneral-regs-only
CFLAGS_REMOVE_nap_nn_neon.o += -mgeneral-regs-only

CFLAGS_nap_fpu.o     += -march=armv8-a+simd
CFLAGS_nap_nn_neon.o += -march=armv8-a+simd
