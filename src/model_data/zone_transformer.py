"""Definition of matrix transformers that area used to
transform (demand, impedance, etc.) matrices between different
zoning systems.
"""
from abc import ABC, abstractmethod

import geopandas as gpd
import numpy as np

try:
    import numpy.typing as npt
except:
    pass
from typing import TypeVar, cast

# bsr_array not supported by older scipy.sparse (Emme)
#from scipy.sparse import bsr_array
from scipy.sparse import bsr_matrix

from model_data.zone_mapping import ZoneMapping

MIN_INTERSECTION_AREA = 0.00000001

DType = TypeVar("DType", bound=np.floating)

def mapping_to_geometry(mapping: ZoneMapping, geometry: str) -> gpd.GeoDataFrame:
    """Gets geometry and area information from zone mapping"""

    result: gpd.GeoDataFrame = gpd.GeoDataFrame(data=mapping.zone_data[geometry],
                              geometry=geometry).reset_index(drop=True) # type: ignore
    result['area'] = result.geometry.area
    result['offset'] = result.index.astype(int)
    return result

class ZoneTransformer(ABC):
    """A class to handle matrix transformations between zone mappings
    """
    @abstractmethod
    def transform_matrix(self, matrix: 'npt.NDArray[np.number]') \
        -> 'npt.NDArray[np.floating]':
        """Abstract base class for matrix transformer that converts matrices
        from one zoning system to an other.

        Args:
            matrix (np.ndarray): matrix in the original zoning system

        Returns:
            np.ndarray: Transformed matrix in the new zoning system
        """
        raise NotImplementedError

    @abstractmethod
    def transform_vectors(self, vector: 'npt.NDArray[np.floating]') \
        -> 'npt.NDArray[np.floating]':
        """Abstract base class for matrix transformer that converts 1D vector
        from one zoning system to an other.

        Args:
            matrix (np.ndarray): vector in the original zoning system

        Returns:
            np.ndarray: Transformed vector in the new zoning system
        """
        raise NotImplementedError


class AreaShareTransformer(ZoneTransformer):
    """A class to transform matrices between zoning systems using
    the intersecting areas between the zones
    """
    
    _multiplier: bsr_matrix
    """Matrix multiplier used for the transformation"""
    
    def __init__(self, origin: ZoneMapping, destination: ZoneMapping):
        """Initializes matrix transformer

        Args:
            origin (ZoneMapping): Zone mapping used for input matrices
            destination (ZoneMapping): Zone mapping used for result matrices
        """
        origin_areas = mapping_to_geometry(origin, 'polygon')
        destination_areas = mapping_to_geometry(destination, 'polygon')
        assert origin_areas.geometry.crs == destination_areas.geometry.crs,\
            'Zoning systems must use the same CRS.'

        
        intersection = cast(gpd.GeoDataFrame,
                gpd.overlay(df1=origin_areas,
                            df2=destination_areas,
                            how='intersection'))
        intersection['area'] = intersection.geometry.area
        intersection = cast(gpd.GeoDataFrame, 
                intersection[(intersection.area > MIN_INTERSECTION_AREA)])

        row = intersection.offset_1.to_numpy()
        col = intersection.offset_2.to_numpy()
        factor = intersection.area / intersection.area_1
        
        self._multiplier = bsr_matrix((factor, (row, col)),
                                     shape=(len(origin_areas.index),
                                            len(destination_areas.index)))
        
        print(self._multiplier)
        
    
    def transform_matrix(self, matrix: 'npt.NDArray[np.number]') \
        -> 'npt.NDArray[np.floating]':
        # [a X a] @ [a X b] -> [a X b]
        a = (matrix @ self._multiplier) 
        b = self._multiplier.transpose() @ a
        return b
    
    def transform_vectors(self, vector: np.ndarray) -> np.ndarray:
        return vector @ self._multiplier
    
    def transform_vectors(self, vector: np.ndarray) -> np.ndarray:
        return vector @ self._multiplier
