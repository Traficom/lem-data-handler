"""Loads Level-of-Service data from individual subarea matrices"""
import time
from pathlib import Path

from model_data.los import (LOSData, LOSMode, LOSType, TimePeriod,
                            get_helmet_matrix_spec)

st = time.process_time()

# Path to the LOS directory
LOS_PATH= Path('./Pohjois-Suomi')

# Load LOS using Helmet naming scheme
los = LOSData.from_omx_files(get_helmet_matrix_spec(LOS_PATH),
                                               mapping=None)

# Get AHT transit time
aht_transit_time = los[TimePeriod.AHT][(LOSMode.TRANSIT_WORK, LOSType.TIME)].to_numpy()
print(aht_transit_time)

# Get LOS for return trip
aht_transit_return_trips = \
    los[TimePeriod.AHT][(LOSMode.TRANSIT_WORK, LOSType.TIME)].reverse_trip()
print(aht_transit_return_trips)

# Define demand distribution and direction for each time period
demand_distribution = {
    TimePeriod.AHT: (0.746026, 0.015065),
    TimePeriod.PT:  (0.234217, 0.329877),
    TimePeriod.IHT: (0.019757, 0.655057),
}
# Calculate demand averaged LOS data
average_los = los.get_averaged_los(demand_distribution)
print(average_los)

# Get averaged transit time
average_transit_time = average_los[(LOSMode.TRANSIT_WORK, LOSType.TIME)].to_numpy()

print(average_transit_time)

del los

print('CPU Execution time:', time.process_time() - st, 'seconds')
