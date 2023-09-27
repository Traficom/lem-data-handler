"""Data loaders for importing data into the model zone mapping"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, cast
try:
    from typing import Protocol
except ImportError:
    # Protocol not supported in older python 3.7 (Emme)
    from typing_extensions import Protocol
import pandas as pd
import geopandas as gpd
from model_data.constants import GEO_ENGINE

from model_data.zone_mapping import ZoneMapping


class ZoneDataType(Enum):
    POPULATION=auto()
    WORKPLACES=auto()
    
class ZoneMappedData(NamedTuple):
    mapping: ZoneMapping
    data: pd.DataFrame
    
def _check_default_values(mapping: ZoneMapping,
                          result: pd.DataFrame,
                          default_values: Dict[str, float] | float | None) \
                              -> pd.DataFrame:
    if set(result.index.values) == set(mapping.zones):
        return result

    if default_values is None:
        raise IndexError(
            'Not all zones could be found in the input data')

    if isinstance(default_values, float):
        return result.reindex(index=mapping.zones, fill_value=default_values)
    
    if set(default_values.keys()) != set(result.columns):
        raise ValueError('Default values do not match the column names.')
    filled_result = result.reindex(index=mapping.zones)
    for c in filled_result.columns:
        filled_result[c] = filled_result[c].fillna(default_values[c])
    return filled_result
    
    
def _unused_name(reserved_strings: Iterable[str], name: str) -> str:
    """Find a distinct name that does not exist in a reserved strings list.

    Args:
        reserved_strings (Iterable[str]): Reserved strings
        name (str): Bases name for new object

    Returns:
        str: New name that does not exist in the reserved strings list
    """
    if name not in reserved_strings:
        return name
    return _unused_name(reserved_strings, '_'+name)

@dataclass
class _IntersectionResult:
    data: pd.DataFrame
    zone_ids: pd.Series
    data_areas: pd.Series
    zone_areas: pd.Series
    intersection_areas: pd.Series
    @property
    def data_area_share(self) -> pd.Series:
        return self.intersection_areas / self.data_areas
    @property
    def zone_area_share(self) -> pd.Series:
        return self.intersection_areas / self.zone_areas


def _get_intersection(input_data: gpd.GeoDataFrame,
                      mapping: ZoneMapping,
                      validate: bool,
                      threshold: float) -> _IntersectionResult:
    """Returns the intersection of GeoDataFrame and ZoneMapping.

    Args:
        data (gpd.GeoDataFrame): Data GeoDataFrame
        mapping (ZoneMapping): Zone mapping
        validate (bool): If set to true the geometry is validated before processing
        threshold: (float): Minimum area to consider intersecting area. 
            Smaller values are discarded

    Raises:
        ValueError: If data or zone mapping contains missing or invalid geometry

    Returns:
        Tuple[pd.DataFrame, str, str, str]: Result as Intersection dataframe
    """
    data_areas = input_data.area
    if validate:
        if not mapping.has_geometry():
            raise ValueError("Used zone mapping has missing or zero area polygons.")
        if (data_areas.isnull().any() or data_areas.isin([None, 0.0]).any()):
            raise ValueError(
                "Given input data has missing or zero area polygon geometries.")
            
    input_area_name = _unused_name(input_data.columns, 'area')
    input_data[input_area_name] = data_areas
    
    zone_data = gpd.GeoDataFrame(data = mapping.zone_data[['polygon']],
                                 geometry='polygon') # type: ignore
    zone_area_name = _unused_name(zone_data.columns, 'zone_area')
    zone_data[zone_area_name] = zone_data.area
    
    intersection = cast(gpd.GeoDataFrame,
                        gpd.overlay(df1=zone_data.reset_index(),
                        df2=input_data,
                        how='intersection',
                        keep_geom_type=False))
    intersection_area_name = _unused_name(intersection.columns, 'intersection_area')
    i_areas = intersection.geometry.area
    intersection[intersection_area_name] = i_areas
    # Drop intersecting areas below threshold and geometry column
    intersection = cast(pd.DataFrame, intersection[i_areas > threshold])\
        .drop(columns='geometry')
    zone_ids = intersection.pop('zone_id')
    input_area = intersection.pop(input_area_name)
    zone_area = intersection.pop(zone_area_name)
    intersection_area = intersection.pop(intersection_area_name)
    return _IntersectionResult(data=intersection,
                               zone_ids=zone_ids,
                               data_areas=input_area,
                               zone_areas=zone_area,
                               intersection_areas=intersection_area)

def _area_share(intersection: _IntersectionResult) -> pd.DataFrame:
    return intersection.data.mul(intersection.data_area_share, axis=0)\
        .groupby(intersection.zone_ids)\
        .sum()
        
def _largest_area(intersection: _IntersectionResult) -> pd.DataFrame:
    zone_id_name = _unused_name(intersection.data.columns, 'zone_id')
    area_name = _unused_name(intersection.data.columns, 'i_areas')
    intersection.data[zone_id_name] = intersection.zone_ids
    intersection.data[area_name] = intersection.intersection_areas
    intersection.data = intersection.data.sort_values(by=[zone_id_name, area_name],
                                                      ascending=[True, False])\
                                         .drop(columns=[area_name])
    grouped = intersection.data.groupby('zone_id').first()
    intersection.data = intersection.data.drop(columns=[zone_id_name])
    return grouped
    

def _intersection_aggregation(mapping: ZoneMapping,
                      data: gpd.GeoDataFrame,
                      default_values: Dict[str, float] | float | None,
                      validate,
                      threshold,
                      aggregation: Callable[[_IntersectionResult], pd.DataFrame]) \
                          -> ZoneMappedData:

    input_geometry = data.geometry.name
    res_columns = list(data.columns)
    res_columns.remove(input_geometry)

    intersection = _get_intersection(data,
                                     mapping,
                                     validate,
                                     threshold)
    result = aggregation(intersection)
    result = _check_default_values(mapping, result, default_values)

    return ZoneMappedData(mapping, result)

def largest_area_aggregation(mapping: ZoneMapping,
                           data: gpd.GeoDataFrame,
                           columns: List[str] | None = None,
                           default_values: Dict[str, float] | float | None = None,
                           validate = True,
                           threshold = 1.0) -> ZoneMappedData:
    return _intersection_aggregation(mapping=mapping,
                                     data=data,
                                     default_values=default_values,
                                     validate=validate,
                                     threshold=threshold,
                                     aggregation=_largest_area)

class Aggregation(Protocol):
    def __call__(self,
             mapping: ZoneMapping,
             data: pd.DataFrame | gpd.GeoDataFrame,
             default_values: Dict[str, float] | None) -> ZoneMappedData:
        ...

class IndexAggregation(Aggregation):
    _index_column: str|None
    """Aggregates zone data based on the index value. Each index is mapped one-to-one 
    to the corresponding zone according to the zone_id."""
    def __init__(self, index_column: str|None = None):
        """Initializes IndexAggregation

        Args:
            index_column (str | None, optional): Column to use as index.
            The specified index column will be dropped from the result dataframe.
            Defaults to None (Use existing DataFrame index).
        """
        self._index_column = index_column
    def __call__(self,
             mapping: ZoneMapping,
             data: pd.DataFrame | gpd.GeoDataFrame,
             default_values: Dict[str, float] | None = None) -> ZoneMappedData:
        """Aggregates zone data based on the index value. Each index is mapped 
        one-to-one to the corresponding zone according to the zone_id.

        Args:
            mapping (ZoneMapping): Destination ZoneMapping for the aggregation
            data (pd.DataFrame): Input data as pandas DataFrame
            columns (List[str] | None, optional): Columns that are expected to be found
                in the data. Defaults to None.
            default_values (Dict[str, float] | None, optional): Default values to use
                when zone id is not found in the input data. Values are given as a dict
                in format {column_name: default_value}. If the parameter is set to None
                or the column name for missing data is not found in the dict, an 
                ValueError will be raised. Defaults to None.

        Raises:
            IndexError: When zone id is not found in the input data and the
                default value is not specified.
            ValueError: When default values do not match the column names.

        Returns:
            ZoneMappedData: ZoneMappedData for the given input 
        """
        result = pd.DataFrame(index=mapping.zones)
            
        merged = pd.merge(result,
                          data,
                          how='inner',
                          left_index=True,
                          right_on=self._index_column,
                          right_index=(self._index_column is None),
                          validate='one_to_one',
                          indicator=False)
        if self._index_column:
            merged = merged.set_index(self._index_column)
            merged.index.name = 'zone_id'
        # Drop geometry column
        merged = merged.drop(columns='geometry', errors='ignore')

        result = _check_default_values(mapping, merged, default_values)
        return ZoneMappedData(mapping, result)
        
class AreaShareAggregation(Aggregation):
    """Aggregates zone data based on the index value. Each index is mapped one-to-one 
    to the corresponding zone according to the zone_id."""
    _threshold: float
    _validate: bool
    def __init__(self, validate: bool = True, area_threshold: float = 1.0):
        """Creates area shere aggregator

        Args:
            validate (bool): If set to true the geometry is validated before processing
            threshold: (float): Minimum area to consider intersecting area. 
                Smaller values are discarded
        """
        self._threshold = area_threshold
        self._validate = validate
    def __call__(self,
                 mapping: ZoneMapping,
                 data: gpd.GeoDataFrame,
                 default_values: Dict[str, float] | None) -> ZoneMappedData:
        return _intersection_aggregation(mapping=mapping,
                                         data=data,
                                         default_values=default_values,
                                         validate=self._validate,
                                         threshold=self._threshold,
                                         aggregation=_area_share)

class LargestAreaAggregation(Aggregation):
    """Aggregates zone data based on the index value. Each index is mapped one-to-one 
    to the corresponding zone according to the zone_id."""
    _threshold: float
    _validate: bool
    def __init__(self, validate: bool = True, area_threshold: float = 1.0):
        """Creates largest area aggregator

        Args:
            validate (bool): If set to true the geometry is validated before processing
            threshold: (float): Minimum area to consider intersecting area. 
                Smaller values are discarded
        """
        self._threshold = area_threshold
        self._validate = validate
    def __call__(self,
                 mapping: ZoneMapping,
                 data: gpd.GeoDataFrame,
                 default_values: Dict[str, float] | None) -> ZoneMappedData:
        return _intersection_aggregation(mapping=mapping,
                                         data=data,
                                         default_values=default_values,
                                         validate=self._validate,
                                         threshold=self._threshold,
                                         aggregation=_largest_area)

def load_data(mapping: ZoneMapping,
                  filepath: Path,
                  aggregation: Aggregation,
                  columns: List[str]|None = None,
                  default_values: Dict[str, float]|None = None,
                  **kwargs) -> ZoneMappedData:
    """Reads GIS data to ZoneMappedData. Uses GeoPandas.read_file and 
    specified Aggregation.

    Args:
        mapping (ZoneMapping): Destination mapping for data.
        filepath (Path): Path to the GIS file
        aggregation (Aggregation): Aggregation function to use
        layer (str | None, optional): Layer name to load. Defaults to None.
        columns (List[str] | None, optional): If specified, only given columns are read.
            Defaults to None (read all columns).
        default_values (Dict[str, float] | None, optional): Default values to use when
            zone id is not found in the input data. Values are given as a dict in
            format {column_name: default_value}. If the parameter is set to None or the
            column name for missing data is not found in the dict, an 
            ValueError will be raised. Defaults to None.

    Returns:
        ZoneMappedData: _description_
    """
    data = cast(gpd.GeoDataFrame, gpd.read_file(filepath,
                                            engine=GEO_ENGINE,
                                            **kwargs))
    #Try convert everything numeric
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='ignore') # type: ignore
        
    if isinstance(data, gpd.GeoDataFrame):
        geometry_columns = [data.geometry.name]
        missing_geometry = data.geometry.isnull().all()
        if geometry_columns and missing_geometry:
            data = data.drop(columns=geometry_columns)
            geometry_columns = []
    else:
        geometry_columns = []
        missing_geometry = True

    

    if columns is not None:
        data = data[columns + geometry_columns]
    return aggregation(mapping, data, default_values) # type: ignore
    