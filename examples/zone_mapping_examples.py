from pathlib import Path
from model_data.zone_mapping import ZoneMapping

mapping = ZoneMapping.from_gpkg(Path('network.gpkg'))
#zone_numbers = [110, 120, 130, 140, 150, 160, 170]
#mapping = ZoneMapping.from_zone_numbers(zone_numbers)
#mapping = ZoneMapping.from_omx(Path('demand_aht.omx'))
print(mapping.zone_data)

# Get number of zones
number_of_zones = len(mapping)
# 7

# Get zone numbers as list
zone_numbers = mapping.zones
# [110, 120, 130, 140, 150, 160, 170]

# Get one zone
zone_120 = mapping.get_zone_by_id(120)
# <model_data.zone_mapping.Zone object>(0/120)

# Get offset in matrices for zone
zone_120_offset = zone_120.offset
# 1

# Get offset of multiple zones
offsets = mapping.zone_ids_to_offsets([110, 130, 150])
# [0, 2, 4]

# Get zone by location
first_zone = mapping.get_zone_by_offset(0)
# <model_data.zone_mapping.Zone object>(0/110)

# Get beeline distance between two zones centroids
distance = zone_120.centroid.distance(first_zone.centroid)
# 1000

# Get distance from one centroid to other cell border
border_distance = zone_120.centroid.distance(first_zone.polygon)
# 500

# Get area of the zone
zone_area = zone_120.polygon.area
# 866025

# Iterate all zones
for zone in mapping:
    print(f'Zone {zone.zone_id} is in position {zone.offset}')
# Zone 110 is in position 0
# Zone 120 is in position 1
# Zone 130 is in position 2
# Zone 140 is in position 3
# Zone 150 is in position 4
# Zone 160 is in position 5
# Zone 170 is in position 6
print('done')