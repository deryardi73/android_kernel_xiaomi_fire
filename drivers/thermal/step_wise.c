/*
 *  step_wise.c - A step-by-step Thermal throttling governor
 *
 *  Copyright (C) 2012 Intel Corp
 *  Copyright (C) 2012 Durgadoss R <durgadoss.r@intel.com>
 *
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 2 of the License.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include <linux/kernel.h>
#include <linux/thermal.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <trace/events/thermal.h>

#include "thermal_core.h"

/*
 * "Gaming boost" tuning (disabled by setting gaming_boost=0):
 *
 *  - Escalation debounce: on THERMAL_TREND_RAISING we normally step the
 *    cooling state up by 1 on every poll that's over the trip point. Under
 *    sustained gaming loads this causes visible perf stutter from short
 *    thermal blips that don't reflect a real sustained trend. We now
 *    require `raise_hold` consecutive over-trip polls (per trip) before
 *    actually stepping up; until then we hold the current state instead of
 *    escalating further. THERMAL_TREND_RAISE_FULL (emergency) and the
 *    initial/uninitialized engage path are left untouched - no debounce
 *    there, those exist for real thermal safety, not perf smoothing.
 *
 *  - Faster recovery: on THERMAL_TREND_DROPPING we normally release the
 *    cooling state by 1 step per poll. That makes perf recovery after a
 *    thermal event feel sluggish. We now release `drop_step` states per
 *    poll instead, so clocks ramp back up quicker once the zone is
 *    actually cooling.
 *
 * Per-zone debounce counters live in tz->governor_data (allocated in
 * step_wise_bind_to_tz()/freed in step_wise_unbind_from_tz()), so no
 * changes to the shared thermal_instance/thermal_zone_device structs are
 * needed.
 */
static bool gaming_boost = true;
module_param(gaming_boost, bool, 0644);
MODULE_PARM_DESC(gaming_boost,
	"reduce throttle stutter and speed up recovery under sustained load (default: Y)");

static unsigned int raise_hold = 2;
module_param(raise_hold, uint, 0644);
MODULE_PARM_DESC(raise_hold,
	"consecutive over-trip polls required before escalating further (gaming_boost only, default: 2)");

static unsigned int drop_step = 2;
module_param(drop_step, uint, 0644);
MODULE_PARM_DESC(drop_step,
	"cooling states released per dropping poll (gaming_boost only, default: 2)");

/* Per-zone private governor state. */
struct step_wise_priv {
	unsigned int trips;		/* tz->trips at bind time */
	unsigned int *raise_count;	/* [0..trips-1]: normal trips
					 * [trips]: forced-passive (THERMAL_TRIPS_NONE)
					 */
};

static inline unsigned int *step_wise_raise_slot(struct thermal_zone_device *tz,
						  int trip)
{
	struct step_wise_priv *priv = tz->governor_data;

	if (!priv || !gaming_boost)
		return NULL;

	if (trip == THERMAL_TRIPS_NONE)
		return &priv->raise_count[priv->trips];

	if (trip < 0 || (unsigned int)trip >= priv->trips)
		return NULL;

	return &priv->raise_count[trip];
}

static int step_wise_bind_to_tz(struct thermal_zone_device *tz)
{
	struct step_wise_priv *priv;

	priv = kzalloc(sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;

	priv->trips = tz->trips;
	/* +1 slot reserved for the forced-passive (THERMAL_TRIPS_NONE) path */
	priv->raise_count = kcalloc(tz->trips + 1, sizeof(*priv->raise_count),
				    GFP_KERNEL);
	if (!priv->raise_count) {
		kfree(priv);
		return -ENOMEM;
	}

	tz->governor_data = priv;

	return 0;
}

static void step_wise_unbind_from_tz(struct thermal_zone_device *tz)
{
	struct step_wise_priv *priv = tz->governor_data;

	if (!priv)
		return;

	kfree(priv->raise_count);
	kfree(priv);
	tz->governor_data = NULL;
}

/*
 * If the temperature is higher than a trip point,
 *    a. if the trend is THERMAL_TREND_RAISING, use higher cooling
 *       state for this trip point
 *    b. if the trend is THERMAL_TREND_DROPPING, do nothing
 *    c. if the trend is THERMAL_TREND_RAISE_FULL, use upper limit
 *       for this trip point
 *    d. if the trend is THERMAL_TREND_DROP_FULL, use lower limit
 *       for this trip point
 * If the temperature is lower than a trip point,
 *    a. if the trend is THERMAL_TREND_RAISING, do nothing
 *    b. if the trend is THERMAL_TREND_DROPPING, use lower cooling
 *       state for this trip point, if the cooling state already
 *       equals lower limit, deactivate the thermal instance
 *    c. if the trend is THERMAL_TREND_RAISE_FULL, do nothing
 *    d. if the trend is THERMAL_TREND_DROP_FULL, use lower limit,
 *       if the cooling state already equals lower limit,
 *       deactivate the thermal instance
 */
static unsigned long get_target_state(struct thermal_zone_device *tz,
				struct thermal_instance *instance,
				enum thermal_trend trend, bool throttle)
{
	struct thermal_cooling_device *cdev = instance->cdev;
	unsigned long cur_state;
	unsigned long next_target;
	unsigned int *raise_cnt;

	/*
	 * We keep this instance the way it is by default.
	 * Otherwise, we use the current state of the
	 * cdev in use to determine the next_target.
	 */
	cdev->ops->get_cur_state(cdev, &cur_state);
	next_target = instance->target;
	dev_dbg(&cdev->device, "cur_state=%ld\n", cur_state);

	if (!instance->initialized) {
		if (throttle) {
			next_target = (cur_state + 1) >= instance->upper ?
					instance->upper :
					((cur_state + 1) < instance->lower ?
					instance->lower : (cur_state + 1));
		} else {
			next_target = THERMAL_NO_TARGET;
		}

		return next_target;
	}

	/* Any trend other than a plain RAISING resets the debounce streak. */
	if (trend != THERMAL_TREND_RAISING) {
		raise_cnt = step_wise_raise_slot(tz, instance->trip);
		if (raise_cnt)
			*raise_cnt = 0;
	}

	switch (trend) {
	case THERMAL_TREND_RAISING:
		if (throttle) {
			raise_cnt = step_wise_raise_slot(tz, instance->trip);

			if (raise_cnt) {
				(*raise_cnt)++;
				if (*raise_cnt < max(raise_hold, 1U)) {
					/*
					 * Absorb a short thermal blip: hold
					 * the current state instead of
					 * escalating on every single poll.
					 */
					next_target = cur_state;
					break;
				}
				*raise_cnt = 0;
			}

			next_target = cur_state < instance->upper ?
				    (cur_state + 1) : instance->upper;
			if (next_target < instance->lower)
				next_target = instance->lower;
		}
		break;
	case THERMAL_TREND_RAISE_FULL:
		if (throttle)
			next_target = instance->upper;
		break;
	case THERMAL_TREND_DROPPING:
		if (cur_state <= instance->lower) {
			if (!throttle)
				next_target = THERMAL_NO_TARGET;
		} else {
			if (!throttle) {
				unsigned int step = gaming_boost ?
						max(drop_step, 1U) : 1U;

				next_target = (cur_state > step) ?
						cur_state - step :
						instance->lower;
				if (next_target > instance->upper)
					next_target = instance->upper;
				if (next_target < instance->lower)
					next_target = instance->lower;
			}
		}
		break;
	case THERMAL_TREND_DROP_FULL:
		if (cur_state == instance->lower) {
			if (!throttle)
				next_target = THERMAL_NO_TARGET;
		} else
			next_target = instance->lower;
		break;
	default:
		break;
	}

	return next_target;
}

static void update_passive_instance(struct thermal_zone_device *tz,
				enum thermal_trip_type type, int value)
{
	/*
	 * If value is +1, activate a passive instance.
	 * If value is -1, deactivate a passive instance.
	 */
	if (type == THERMAL_TRIP_PASSIVE || type == THERMAL_TRIPS_NONE)
		tz->passive += value;
}

static void thermal_zone_trip_update(struct thermal_zone_device *tz, int trip)
{
	int trip_temp;
	enum thermal_trip_type trip_type;
	enum thermal_trend trend;
	struct thermal_instance *instance;
	bool throttle = false;
	int old_target;

	if (trip == THERMAL_TRIPS_NONE) {
		trip_temp = tz->forced_passive;
		trip_type = THERMAL_TRIPS_NONE;
	} else {
		tz->ops->get_trip_temp(tz, trip, &trip_temp);
		tz->ops->get_trip_type(tz, trip, &trip_type);
	}

	trend = get_tz_trend(tz, trip);

	if (tz->temperature >= trip_temp) {
		throttle = true;
		trace_thermal_zone_trip(tz, trip, trip_type);
	}

	dev_dbg(&tz->device, "Trip%d[type=%d,temp=%d]:trend=%d,throttle=%d\n",
				trip, trip_type, trip_temp, trend, throttle);

	mutex_lock(&tz->lock);

	list_for_each_entry(instance, &tz->thermal_instances, tz_node) {
		if (instance->trip != trip)
			continue;

		old_target = instance->target;
		instance->target = get_target_state(tz, instance, trend, throttle);
		dev_dbg(&instance->cdev->device, "old_target=%d, target=%d\n",
					old_target, (int)instance->target);

		if (instance->initialized && old_target == instance->target)
			continue;

		/* Activate a passive thermal instance */
		if (old_target == THERMAL_NO_TARGET &&
			instance->target != THERMAL_NO_TARGET)
			update_passive_instance(tz, trip_type, 1);
		/* Deactivate a passive thermal instance */
		else if (old_target != THERMAL_NO_TARGET &&
			instance->target == THERMAL_NO_TARGET)
			update_passive_instance(tz, trip_type, -1);

		instance->initialized = true;
		mutex_lock(&instance->cdev->lock);
		instance->cdev->updated = false; /* cdev needs update */
		mutex_unlock(&instance->cdev->lock);
	}

	mutex_unlock(&tz->lock);
}

/**
 * step_wise_throttle - throttles devices associated with the given zone
 * @tz - thermal_zone_device
 * @trip - trip point index
 *
 * Throttling Logic: This uses the trend of the thermal zone to throttle.
 * If the thermal zone is 'heating up' this throttles all the cooling
 * devices associated with the zone and its particular trip point, by one
 * step. If the zone is 'cooling down' it brings back the performance of
 * the devices by one step.
 */
static int step_wise_throttle(struct thermal_zone_device *tz, int trip)
{
	struct thermal_instance *instance;

	thermal_zone_trip_update(tz, trip);

	if (tz->forced_passive)
		thermal_zone_trip_update(tz, THERMAL_TRIPS_NONE);

	mutex_lock(&tz->lock);

	list_for_each_entry(instance, &tz->thermal_instances, tz_node)
		thermal_cdev_update(instance->cdev);

	mutex_unlock(&tz->lock);

	return 0;
}

static struct thermal_governor thermal_gov_step_wise = {
	.name		= "step_wise",
	.bind_to_tz	= step_wise_bind_to_tz,
	.unbind_from_tz	= step_wise_unbind_from_tz,
	.throttle	= step_wise_throttle,
};

int thermal_gov_step_wise_register(void)
{
	return thermal_register_governor(&thermal_gov_step_wise);
}

void thermal_gov_step_wise_unregister(void)
{
	thermal_unregister_governor(&thermal_gov_step_wise);
}