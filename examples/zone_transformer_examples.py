from pathlib import Path
import numpy as np
from model_data.zone_mapping import ZoneMapping
from model_data.zone_transformer import AreaShareTransformer

# Define sample data
input_vector = np.ones(shape=7)
input_matrix = np.ones(shape=(7,7))

# Generate source and destination mappings
mapping1 = ZoneMapping.from_gpkg(Path('network.gpkg'))
mapping2 = ZoneMapping.from_gpkg(Path('network2.gpkg'))

# Create transformer
transformer = AreaShareTransformer(mapping1, mapping2)

# Share vector values based on intersecting area
output_vector = transformer.transform_vectors(input_vector)
# [1.0, 0.5, 0.5, 0.5, 1.0, 0.0]

# Distribute matrix values based on intersecting area
output_matrix = transformer.transform_matrix(input_matrix)
# [[1.  0.5  0.5  0.5  1.   0. ]
# [0.5  0.2  0.3  0.2  0.5  0. ]
# [0.5  0.3  0.3  0.3  0.5  0. ]
# [0.5  0.2  0.3  0.2  0.5  0. ]
# [1.   0.5  0.5  0.5  1.   0. ]
# [0.   0.   0.   0.   0.   0. ]]
print('done')