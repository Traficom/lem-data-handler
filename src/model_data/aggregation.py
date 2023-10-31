from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, NamedTuple, cast
try:
    from typing import Protocol
except ImportError:
    # Protocol not supported in older python 3.7 (Emme)
    from typing_extensions import Protocol
import pandas as pd
import geopandas as gpd
from model_data.data_item import DataItem
from model_data.zone_mapping import ZoneMapping

class ZoneMappedData(NamedTuple):
    """Container for model data mapped to a target ZoneMapping"""
    mapping: ZoneMapping
    data: pd.DataFrame
    
def apply_default_values(mapped_data: ZoneMappedData,
                          default_values: Dict[str, float|DataItem] \
                              | float | DataItem | None) -> pd.DataFrame:
    """Checks data for missing values and replaces them with defaults

    Args:
        mapping (ZoneMapping): Target ZoneMapping
        result (pd.DataFrame): Aggregated zone data
        default_values (Dict[str, float | DataItem]): Default values

    Raises:
        IndexError: When not all zones could be found in input data and default values 
            are not specified
        ValueError: When the specified default value names do not match the data columns

    Returns:
        pd.DataFrame: Data where missing values area replaced with defaults.
    """
    result = mapped_data.data
    mapping = mapped_data.mapping
    if set(result.index.values) == set(mapping.zones):
        return result

    if default_values is None:
        raise IndexError(
            'Not all zones could be found in the input data')

    if isinstance(default_values, (float, DataItem)):
        return result.reindex(index=mapping.zones, fill_value=default_values)
    
    if set(default_values.keys()) != set(result.columns):
        raise ValueError('Default values do not match the column names.')
    filled_result = result.reindex(index=mapping.zones)
    for c in filled_result.columns:
        filled_result[c] = filled_result[c].fillna(default_values[c])
    return ZoneMappedData(mapping, filled_result)
    
    
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
    """Representation of geometry intersection results for the helper functions"""
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
    data_areas: pd.Series[float] = input_data.area
    if validate:
        if not mapping.has_geometry():
            raise ValueError("Used zone mapping has missing or zero area polygons.")
        if (data_areas.isnull().any() or data_areas.isin([None, 0.0]).any()):
            raise ValueError(
                "Given input data has missing or zero area polygon geometries.")
            
    data_areas.name = _unused_name(input_data.columns, 'area')
    
    zone_data = gpd.GeoDataFrame(data = mapping.zone_data[['polygon']],
                                 geometry='polygon') # type: ignore
    zone_area_name = _unused_name(zone_data.columns, 'zone_area')
    zone_data[zone_area_name] = zone_data.area
    
    intersection = cast(gpd.GeoDataFrame,
                        gpd.overlay(df1=zone_data.reset_index(),
                        df2=pd.concat([input_data, data_areas.to_frame()], axis=1),
                        how='intersection',
                        keep_geom_type=False))
    intersection_area_name = _unused_name(intersection.columns, 'intersection_area')
    i_areas = intersection.geometry.area
    intersection[intersection_area_name] = i_areas
    # Drop intersecting areas below threshold and geometry column
    intersection = cast(pd.DataFrame, intersection[i_areas > threshold])\
        .drop(columns='geometry')
    zone_ids = intersection.pop('zone_id')
    input_area = intersection.pop(data_areas.name)
    zone_area = intersection.pop(zone_area_name)
    intersection_area = intersection.pop(intersection_area_name)
    return _IntersectionResult(data=intersection,
                               zone_ids=zone_ids,
                               data_areas=input_area,
                               zone_areas=zone_area,
                               intersection_areas=intersection_area)

def _area_share(intersection: _IntersectionResult) -> pd.DataFrame:
    """Intersection aggregation helper that splits the input data
    in the proportions of the intersecting areas

    Args:
        intersection (_IntersectionResult): _IntersectionResult to process

    Returns:
        pd.DataFrame: Aggregated dataframe
    """
    return intersection.data.mul(intersection.data_area_share, axis=0)\
        .groupby(intersection.zone_ids)\
        .sum(numeric_only=False)
        
def _largest_area(intersection: _IntersectionResult) -> pd.DataFrame:
    """Intersection aggregation helper that picks the data from the
        largest intersecting area

    Args:
        intersection (_IntersectionResult): _IntersectionResult to process

    Returns:
        pd.DataFrame: Aggregated dataframe
    """
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
    return ZoneMappedData(mapping, aggregation(intersection))

class Aggregation(Protocol):
    """Protocol definition for area aggregation functions"""
    def __call__(self,
             mapping: ZoneMapping,
             data: pd.DataFrame | gpd.GeoDataFrame) -> ZoneMappedData:
        """Aggregates the input data to the specified ZoneMapping.

        Args:
            mapping (ZoneMapping): Target ZoneMapping
            data (pd.DataFrame | gpd.GeoDataFrame): Input data.
            zones not found in the input data. Can be specified either a single value
            to be used for all columns or dict of column-name value pairs.

        Returns:
            ZoneMappedData: Input data mapped to the target ZoneMapping
        """
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
             data: pd.DataFrame | gpd.GeoDataFrame) -> ZoneMappedData:
        """Aggregates zone data based on the index value. Each index is mapped 
        one-to-one to the corresponding zone according to the zone_id.

        Args:
            mapping (ZoneMapping): Destination ZoneMapping for the aggregation
            data (pd.DataFrame): Input data as pandas DataFrame
            columns (List[str] | None, optional): Columns that are expected to be found
                in the data. Defaults to None.

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

        return ZoneMappedData(mapping, merged)
        
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
                 data: gpd.GeoDataFrame) -> ZoneMappedData:
        return _intersection_aggregation(mapping=mapping,
                                         data=data,
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
                 data: gpd.GeoDataFrame) -> ZoneMappedData:
        return _intersection_aggregation(mapping=mapping,
                                         data=data,
                                         validate=self._validate,
                                         threshold=self._threshold,
                                         aggregation=_largest_area)