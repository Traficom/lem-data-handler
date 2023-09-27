"""Trip impedance processing"""
from __future__ import annotations
from enum import Enum, auto
import lzma
import pickle
from tables.carray import CArray

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, cast

from model_data.zone_mapping import ZoneMapping, IncompatibleZoneException

import numpy as np
import numpy.typing as npt
import openmatrix as omx

logger = logging.getLogger(__name__)

LOSDtype = np.float16
LOS_MISSING =  np.finfo(LOSDtype).max


class OMXSpecification(NamedTuple):
    """Specification of OMX file layout"""
    file: Path
    """Path to the OMX file"""
    matrix: str
    """Name of the matrix object in the file"""

def _replace_submatrix(matrix: np.ndarray, zones: List[int],
        submatrix: np.ndarray, subzones: List[int]):
    """Replaces the selected area of the first input matrix with the specified
        submatrix inplace

    Args:
        matrix (np.ndarray): Larger inputmatrix that will be modified
        zones (List[int]): Zone numbering of the larger matrix
        submatrix (np.ndarray): Smaller submatrix
        subzones (List[int]): Zone numbering in the smaller matrix

    Returns:
        np.ndarray: Numpy array containing the combined matrix.
    """
    matching_indices = np.isin(zones, subzones, assume_unique=True)
    if sum(matching_indices) != len(subzones):
        raise IncompatibleZoneException(
            "Subarea zones do not match the target mapping.")

    matrix[np.ix_(matching_indices,matching_indices)] = submatrix

def get_helmet_matrix_spec(base_dir: Path) -> Dict[TimePeriod, Dict[LOSType, 
                                                        OMXSpecification]]:
    """Returns the OMX matrix naming scheme for Helmet model system

    Args:
        base_dir (Path): Directory for the OMX files

    Returns:
        Dict[TimePeriod, Dict[LOSType, OMXSpecification]]: OMX file specs for 
            AHT, PT and IHT time periods
    """
    time_periods = {
        TimePeriod.AHT: 'aht',
        TimePeriod.PT: 'pt',
        TimePeriod.IHT: 'iht'        
    }

    los_types = {
        LOSType.CAR_COST: ('cost', 'car_work'),
        LOSType.CAR_DISTANCE: ('dist', 'car_work'),
        LOSType.CAR_TIME: ('time', 'car_work'),
        LOSType.TRANSIT_COST: ('cost', 'transit_work'),
        LOSType.TRANSIT_TIME: ('time', 'transit_work')
    }
    
    return dict(
        [(time_period, 
          dict([(los_type, 
                 OMXSpecification(base_dir / (los_types[los_type][0] + '_' + \
                     time_periods[time_period] + '.omx'), 
                                  los_types[los_type][1]))
                for los_type in LOSType]))
          for time_period in TimePeriod])


class LOSMatrix:
    """Representation of a single Level-of-Service matrix"""
    _mapping: ZoneMapping
    """Zone mapping the matrix is using"""
    _data: npt.NDArray[LOSDtype]
    """Matrix data"""
    
    @classmethod
    def zeros_like(cls, other: LOSMatrix):
        """Builds new LOSMatrix with identical parameters to the specified matrix
        but with all values zeroed."""
        return LOSMatrix(other._mapping, np.zeros_like(other.to_numpy()))
    
    @classmethod
    def from_omx(cls,
                 omx_spec: OMXSpecification,
                 mapping: ZoneMapping | None = None,
                 validate: bool = True) -> LOSMatrix:
        """Reads LOSMatrix from OMX file

        Args:
            filename (Path): Path to the OMX file
            mapping (ZoneMapping): Zone mapping to use with the matrix. If None, the 
                mapping will be automatically generated from the OMX file
            matrix_name (str): Name of the matrix in the file
            validate (bool, optional): If set to True the zone numbers area validated 
                against the ZoneMapping. Defaults to True.

        Returns:
            LOSMatrix: LOS matrix with the data loaded from the OMX file
        """
        
        file = omx.open_file(omx_spec.file, mode='r')
        matrix = np.array(file[omx_spec.matrix]).astype(LOSDtype)
        zones_in_file = file.map_entries('zone_number')
        file.close()
        if mapping is None:
            # Create mapping from OMX file
            mapping = ZoneMapping.from_omx(omx_spec.file)
        elif validate and (zones_in_file != mapping.zones):
            raise IncompatibleZoneException(
                'zone_numbers in OMX file do not match the used zone mapping')
        return LOSMatrix(mapping, matrix)

    @classmethod
    def from_subareas(cls,
                        mapping: ZoneMapping,
                        subareas: List[LOSMatrix],
                        default_value: LOSDtype) -> LOSMatrix:
        """Builds LOS matrix from a set of subarea matrices. The values outside the
        

        Args:
            mapping (ZoneMapping): Zone mapping for destination matrix
            subareas (List[LOSMatrix]): A list of submatrices
            default_value (ImpedanceDtype): Constant impedance value to use outside the
                subareas

        Returns:
            LOSMatrix: LOS matrix in the destination zone mapping
        """
        
        data: npt.NDArray[LOSDtype] = np.full(
            shape=(len(mapping), len(mapping)),
            fill_value=default_value,
            dtype=LOSDtype)
        
        target_zones = mapping.zones
        for subarea in subareas:
            source_zones = subarea._mapping.zones
            _replace_submatrix(data,
                               target_zones,
                               subarea.to_numpy(),
                               source_zones)
        return LOSMatrix(mapping, data)

        
    def __init__(self, mapping: ZoneMapping, data: npt.NDArray[LOSDtype]):
        if not (data.shape[0] == data.shape[1] == len(mapping)):
            raise IncompatibleZoneException(
                'Number of zones in the matrix does not match the zone mapping')
        self._mapping = mapping
        self._data = data

    def to_omx(self, omx_spec: OMXSpecification) -> None:
        """Writes LOSMatris into OMX file

        Args:
            omx_spec (OMXSpecification): OMX file and matrix name
        """
        
        file = omx.open_file(omx_spec.file, mode='a')
        file.create_mapping('zone_number',
                            entries = self._mapping.zones,
                            overwrite=True)
        mat_name = omx_spec.matrix
        if mat_name in file:
            mat = cast(CArray, file[mat_name])
            mat[...] = self._data
        else:
            mat = file.create_matrix(name=mat_name, obj=self._data)
        file.close()
    
    def to_numpy(self) -> npt.NDArray[LOSDtype]:
        """Returns the LOS data as numpy array.

        Returns:
            npt.NDArray[ImpedanceDtype]: Numpy array containeing the LOS data.
        """
        return self._data
    
    def reverse_trip(self) -> npt.NDArray[LOSDtype]:
        """Returns the LOS data for the reversed trips as numpy array.

        Returns:
            npt.NDArray[ImpedanceDtype]: Numpy array containeing the LOS data for
                the reverse trips.
        """
        return self._data.T
    
    def direction_weighted_sum(self,
                               weights: Tuple[float, float]) -> LOSMatrix:
        """Calculates the weighted sum of the forward and reverse trips

        Args:
            weights (Tuple[float, float]): Weights for the directions (forward, reverse)

        Returns:
            LOSMatrix: Weighted sum of forward and reverse directions
        """
        return LOSMatrix(self._mapping,
                         self.to_numpy() * weights[0] \
                             + self.reverse_trip() * weights[1])
        
    
    def __add__(self, other: LOSMatrix):
        """Adds two LOSMatrix together

        Args:
            other (LOSMatrix): Other LOSMatrix to add

        Returns:
            _type_: The sum of two matrices
        """
        assert len(self._mapping) == len(other._mapping), \
            "Added LOSMatrices must have the same number of zones"
        return LOSMatrix(self._mapping, self.to_numpy() + other.to_numpy())

    def __iadd__(self, other: LOSMatrix):
        """Increments the object with other LOSMatrix

        Args:
            other (LOSMatrix): Other LOSMatrix to add

        Returns:
            _type_: Self incremented by the other object.
        """
        assert len(self._mapping) == len(other._mapping), \
            "Added LOSMatrices must have the same number of zones"
        self._data += other._data
        return self

class TimePeriod(Enum):
    """Assignemtn time periods"""
    AHT=auto()
    PT=auto()
    IHT=auto()
    
class LOSType(Enum):
    """Level-of-service types"""
    CAR_TIME=auto()
    CAR_DISTANCE=auto()
    CAR_COST=auto()
    TRANSIT_TIME=auto()
    TRANSIT_COST=auto()

class LOSTimePeriod:
    """Container for various level-of-service matrices"""
    _matrices: Dict[LOSType, LOSMatrix]

    @classmethod
    def zeros_like(cls, other: LOSTimePeriod) -> LOSTimePeriod:
        """Returns an identical LOSTimePeriod to the specified object with
        all the matrices zeroed.

        Args:
            other (LOSTimePeriod): Other LOSTimePeriod where the information
            is copied from

        Returns:
            LOSTimePeriod: Identical LOSTimePeriod to the specified object with
        all the matrices zeroed.
        """
        new_matrices = {}
        for key, value in other._matrices.items():
            new_matrices[key] = LOSMatrix.zeros_like(value)
        return LOSTimePeriod(new_matrices)
    
    
    def __init__(self, los_matrices: Dict[LOSType, LOSMatrix]):
        self._matrices = los_matrices
    
    def __getitem__(self, type: LOSType) -> LOSMatrix:
        """Returns the LOSMatrix for the specified LOS type

        Args:
            type (LOSType): Level-of-service type

        Returns:
            _type_: Level-of-service matrix for the specified type
        """
        return self._matrices[type]

    def __add__(self, other: LOSTimePeriod):
        """Adds two LOSTimePeriods together

        Args:
            other (LOSTimePeriod): Other LOSTimePeriod to add

        Returns:
            _type_: Sum of two specified objects.
        """
        result = {}
        for key, value in self._matrices.items():
            result[key] = other._matrices[key] + value
        return LOSTimePeriod(result)
    
    def __iadd__(self, other: LOSTimePeriod):
        """Adds two other LOSTimeperiod to this object

        Args:
            other (LOSTimePeriod): Other LOSTimePeriod to add

        Returns:
            _type_: Self incremented with the other object.
        """
        for key in self._matrices.keys():
            self._matrices[key] += other._matrices[key]
        return self
    

class LOSData:

    @classmethod
    def from_subareas(cls, 
                      mapping: ZoneMapping,
                      subareas: List[LOSData],
                      default_values: LOSDtype | Dict[LOSType, LOSDtype]) -> LOSData:
        """Builds new LOS data from subarea LOS matrices.

        Args:
            mapping (ZoneMapping): Destination zone mapping
            subareas (List[LOSData]): List of subarea LOS data
            default_values (LOSDtype | Dict[LOSType, LOSDtype]): Default value for 
                matrix locations not found in the subareas. Can be either single value
                for all LOSTypes or dict of {LOSType: value} pairs.

        Returns:
            LOSData: New LOS Data constructed from the subareas.
        """
        los_periods: Dict[TimePeriod, LOSTimePeriod] = {}
        for time_period in TimePeriod:
            los_matrices: Dict[LOSType, LOSMatrix] = {}
            for los_type in LOSType:
                submatrices = [sub[time_period][los_type] for sub in subareas]
                if isinstance(default_values, Dict):
                    default_value = default_values[los_type]
                else:
                    default_value = default_values
                los_matrices[los_type] = LOSMatrix.from_subareas(mapping, 
                                                                submatrices,
                                                                default_value)
            los_periods[time_period] = LOSTimePeriod(los_matrices)
        return LOSData(los_periods)
    
    @classmethod
    def from_omx_files(cls,
                       omx_specs: Dict[TimePeriod, Dict[LOSType, 
                                                        OMXSpecification]],
                       mapping: ZoneMapping | None = None) -> LOSData:
        """Loads LOS data from OMX files.

        Args:
            omx_specs (Dict[TimePeriod, Dict[LOSType, OMXSpecification]]): OMX file
            
            mapping (ZoneMapping | None, optional): Zone mapping to use. If specified 
                the mapping will be used to validate the zone numbers in OMX file.
                Defaults to None (no checking).

        Returns:
            LOSData: New LOS data from the OMX files.
        """
        los_periods: Dict[TimePeriod, LOSTimePeriod] = {}
        for time_period in TimePeriod:
            los_matrices: Dict[LOSType, LOSMatrix] = {}
            for los_type in LOSType:
                omx_spec = omx_specs[time_period][los_type]
                los_matrices[los_type] = LOSMatrix.from_omx(omx_spec,
                                                            mapping)
            los_periods[time_period] = LOSTimePeriod(los_matrices)
        return LOSData(los_periods)
    
    @classmethod
    def from_pickle(cls, filename: Path) -> LOSData:
        """Loads the LOS data from LZMA compressed pickle.

        Args:
            filename (Path): Path to the pickle file.

        Returns:
            LOSData: New LOS data loaded from the file.
        """
        data = pickle.load(lzma.open(filename, 'r'))
        return data
        

    _data: Dict[TimePeriod, LOSTimePeriod]
    
    def __init__(self, lost_time_periods: Dict[TimePeriod, LOSTimePeriod]):
        self._data = lost_time_periods
    

    def to_omx_files(self,
                     omx_specs: Dict[TimePeriod, Dict[LOSType, 
                                                        OMXSpecification]]) -> None:
        """Writes Level-of-Service data into OMX files according the given naming scehme

        Args:
            omx_specs (Dict[TimePeriod, Dict[LOSType, OMXSpecification]]): OMX naming 
                scheme
        """
        for time_period in TimePeriod:
            for los_type in LOSType:
                omx_spec = omx_specs[time_period][los_type]
                self._data[time_period][los_type].to_omx(omx_spec)
    
    def to_pickle(self, filename: Path) -> None:
        """Saves LOS data as LZMA compressed pickle.

        Args:
            filename (Path): Path to the destination file.
        """
        pickle.dump(self, lzma.open(filename, 'wb'))

    def __getitem__(self, time_period: TimePeriod) -> LOSTimePeriod:
        """Returns the LOSMatrix for the specified LOS type

        Args:
            type (LOSType): Level-of-service type

        Returns:
            _type_: Level-of-service matrix for the specified type
        """
        return self._data[time_period]

    def get_averaged_los(self,
                     weights: Dict[TimePeriod, Tuple[float, float]]) -> LOSTimePeriod:
        """Returns and averaged LOS over all the time periods and directions.

        Args:
            weights (Dict[TimePeriod, Tuple[float, float]]): Dictionary of time period 
                weights. Each weight is a tuple of (forward, reverse) direction weights.

        Returns:
            LOSTimePeriod: Weighted sum of the Level of Service.
        """
        result = LOSTimePeriod.zeros_like(self._data[TimePeriod.AHT])
        for time_period, weight in weights.items():
            for los_type in result._matrices.keys():
                result._matrices[los_type] \
                    += self._data[time_period][los_type].direction_weighted_sum(weight)
        return result
