"""Loads Level-of-Service data from individual subarea matrices"""
from pathlib import Path
from typing import List
from model_data.los import LOS_MISSING, LOSData, get_helmet_matrix_spec, LOSType, TimePeriod
from model_data.zone_mapping import ZoneMapping


# Path to the LOS directory
LOS_PATH= Path('./los')

# Load LOS using Helmet naming scheme
los = LOSData.from_omx_files(get_helmet_matrix_spec(LOS_PATH),
                                               mapping=None)

# Get AHT transit time
aht_transit_time = los[TimePeriod.AHT][LOSType.TRANSIT_TIME].to_numpy()
print(aht_transit_time)

# Get LOS for return trip
aht_transit_return_trips = los[TimePeriod.AHT][LOSType.TRANSIT_TIME].reverse_trip()
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
average_transit_time = average_los[LOSType.TRANSIT_TIME].to_numpy()

print(average_transit_time)