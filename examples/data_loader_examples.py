from pathlib import Path
from model_data.data_loader import (load_data, IndexAggregation, AreaShareAggregation,
                                    LargestAreaAggregation)
from model_data.zone_mapping import ZoneMapping

defaults = {'a': 0., 'b':999.}
mapping = ZoneMapping.from_gpkg(Path('network.gpkg'))

# Use IndexAggregation
zone_data = load_data(mapping = mapping, 
                      filepath = Path('zone_data.csv'),
                      aggregation = IndexAggregation('zone'),
                      default_values = defaults)

# Use AreaShareAggregation
zone_data = load_data(mapping = mapping, 
                      filepath = Path('zone_data.gpkg'),
                      layer = 'polygon_data',
                      aggregation = AreaShareAggregation(),
                      default_values = defaults)

# Use LargestAreaAggregation
zone_data = load_data(mapping = mapping, 
                      filepath = Path('zone_data.gpkg'),
                      layer = 'polygon_data',
                      aggregation = LargestAreaAggregation(),
                      default_values = defaults)
