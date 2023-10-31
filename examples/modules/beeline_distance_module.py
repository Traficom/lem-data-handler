"""Model system module that loads the """
from pathlib import Path
import time
import numpy as np
from typing import Dict
from model_data.module import Module, CachedValue, Phase
from model_data.zone_mapping import ZoneMapping

class BeelineDistanceModule(Module):
    def __init__(self):
        super().__init__(provides=['beeline_distance'])
    
    def process(self, zone_mapping: ZoneMapping) -> Dict[str, CachedValue]:
        if not zone_mapping.has_geometry():
            raise ValueError('Unable to calculate beeline distances.'\
                'The zone mapping does not have valid geometries')
        
        centroids = [zone.centroid for zone in zone_mapping]
        distance = np.zeros(shape=(len(centroids), len(centroids)), dtype=np.float16 )
        st = time.process_time()
        cent = zone_mapping.centroids
        for idx, centroid in enumerate(centroids[:-1]):
            distance[idx, idx:] = cent.iloc[idx:].distance(centroid, align=False)/1000.0
        return {'beeline_distance': CachedValue(value=distance + distance.T,
                                                scope=Phase.SCENARIO)}
    