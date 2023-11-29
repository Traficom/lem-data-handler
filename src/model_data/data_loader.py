"""Data loaders for importing data into the model zone mapping"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple, cast

try:
    from typing import Protocol
except ImportError:
    # Protocol not supported in older python 3.7 (Emme)
    from typing_extensions import Protocol

import geopandas as gpd
import pandas as pd

from model_data.aggregation import (Aggregation, AreaShareAggregation,
                                    IndexAggregation, LargestAreaAggregation,
                                    ZoneMappedData, apply_default_values)
from model_data.constants import GEO_ENGINE
from model_data.data_item import DataItem
from model_data.zone_mapping import ZoneMapping


def load_data(mapping: ZoneMapping,
                  filepath: Path,
                  aggregation: Aggregation,
                  columns: List[str]|None = None,
                  default_values: Dict[str, float|DataItem]|None = None,
                  **kwargs) -> ZoneMappedData:
    """Reads GIS data to ZoneMappedData. Uses GeoPandas.read_file and 
    specified Aggregation.

    Args:
        mapping (ZoneMapping): Destination mapping for data.
        filepath (Path): Path to the GIS file
        aggregation (Aggregation): Aggregation function to use
        columns (List[str] | None, optional): If specified, only given columns are read.
            Defaults to None (read all columns).
        default_values (Dict[str, float] | None, optional): Default values to use when
            zone id is not found in the input data. Values are given as a dict in
            format {column_name: default_value}. If the parameter is set to None or the
            column name for missing data is not found in the dict, an 
            ValueError will be raised. Defaults to None.

    Returns:
        ZoneMappedData: Input data mapped to the ZoneMapping
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
    res = ZoneMappedData(mapping, aggregation(mapping, data))
    if default_values:
        res = apply_default_values(res, default_values)

AGGREGATION = {
    'INDEX': lambda conf: IndexAggregation(conf.get('index_col')),
    'AREA_SHARE': lambda conf: AreaShareAggregation(),
    'LARGEST_AREA': lambda conf: LargestAreaAggregation()
}

class CollisionHandler(Protocol):
    def __call__(self, old_series: pd.Series, new_series: pd.Series) -> pd.Series:
        ...

class RaiseCollisionHandler(CollisionHandler):
    def __call__(self, old_series: pd.Series, new_series: pd.Series) -> pd.Series:
        msg = (f'Duplicate configuration for zone data {old_series.name}. Remove the '
            'extra data definition or use collision rules to combine the data.')
        raise RuntimeError(msg)

class ReplaceCollisionHandler(CollisionHandler):
    def __call__(self, old_series: pd.Series, new_series: pd.Series) -> pd.Series:
        result = pd.Series(old_series)
        result.update(new_series)
        return result
class AddCollisionHandler(CollisionHandler):
    def __call__(self, old_series: pd.Series, new_series: pd.Series) -> pd.Series:
        return old_series + new_series
class ColumnConfig(NamedTuple):
    result: str
    column: str
    shares: Dict[str, str]|None
    collision_handler: CollisionHandler
    
class DataFileConfig(NamedTuple):
    data_file: Path
    extra_arguments: Dict[str, Any]
    aggregation: Aggregation
    columns: List[ColumnConfig]

def combine_zone_data(mapped_data: Iterable[Tuple[ZoneMappedData, 
                                                  Dict[str, CollisionHandler]]]
                      )-> ZoneMappedData:
    """Combines multiple ZoneMappedData using specified collision handlers if the same
    column is specified in multiple data sets.

    Args:
        mapped_data (Iterable[Tuple[ZoneMappedData, Dict[str, CollisionHandler]]]): 
            Data and collision handlers.

    Returns:
        ZoneMappedData: Single ZoneMappedData with the combined data.
    """
    if len(mapped_data) < 1:
        return mapped_data[0][0]
    first: ZoneMappedData = mapped_data[0][0]
    rest: Iterable[Tuple[ZoneMappedData, Dict[str, CollisionHandler]]] = mapped_data[1:]
    assert all([first.mapping == r[0].mapping for r in rest]), \
        "Trying to combine data with different zone mappings"
    result = pd.DataFrame(first.data)
    for new_data, on_collision in rest:
        for col_name, col_values in new_data.data.items():
            result[col_name] = on_collision[col_name](result[col_name], col_values) \
                if col_name in result else col_values
    return ZoneMappedData(first.mapping, result)

def load_files(configs: Iterable[DataFileConfig],
               zone_mapping: ZoneMapping) -> ZoneMappedData:
    """Loads zone data from data files using given configurations

    Args:
        configs (Iterable[DataFileConfig]): File and column configuration
        zone_mapping (ZoneMapping): Destination zone mapping for data

    Returns:
        ZoneMappedData: Loaded data with aggregation and collision handling applied.
    """
    loaded_data :List[Tuple[ZoneMappedData, CollisionHandler]] = []

    for file_config in configs:
        data: gpd.GeoDataFrame = gpd.read_file(filename=file_config.data_file,
                                               **file_config.extra_arguments)
        results = gpd.GeoDataFrame(index=data.index, geometry=data.geometry)
        on_collision = {}
        for col in file_config.columns:
            on_collision[col.result] = col.collision_handler
            if col.shares is not None:
                results[col.result] = data.apply(lambda x:
                    DataItem(total = x[col.column],
                             category_proportions=dict([(k, x[v]) for k, v in
                                                        (col.shares.items()
                                                         if col.shares
                                                         else (None, None))])),
                    axis=1)
            else:
                results[col.result] = data[col.column].astype(float)
        if results.geometry is not None and data.geometry.isnull().all():
            results.pop(results.geometry.name)
        loaded_data.append(( file_config.aggregation(zone_mapping, results),
                             on_collision ))
    return combine_zone_data(loaded_data)

def _get_aggregation(conf: Dict[str, Any]) -> Aggregation:
    """Helper funcion to convert config to Aggregation class

    Args:
        conf (Dict[str, Any]): Config section

    Returns:
        Aggregation: Aggregation function from the config definition
    """
    AGGREGATIONS = {
        'INDEX': lambda x: IndexAggregation(index_column=x.get('index_column', None)),
        'AREA_SHARE': lambda x: AreaShareAggregation(),
        'LARGEST_AREA': lambda x: LargestAreaAggregation(),
    }
    return AGGREGATIONS[conf['aggregation']](conf)

def _get_collision_handler(conf: Dict[str, Any]) -> CollisionHandler:
    """Helper function for converting config to CollisionHandler

    Args:
        conf (Dict[str, Any]): Config section

    Returns:
        CollisionHandler: CollisionHandler from the config
    """
    COLLISION = {
        'RAISE': RaiseCollisionHandler(),
        'REPLACE': ReplaceCollisionHandler(),
        'ADD': AddCollisionHandler(),
    }
    return COLLISION[conf.get('on_collision', 'RAISE')]


def section_to_config(file_conf: Dict[str, Any],
                      base_dir: Path = Path('.')) -> DataFileConfig:
    """Helper function for parsing the JSON configuration

    Args:
        file_conf (Dict[str, Any]): Datafile configuration section

    Returns:
        ZoneDataConfig: ZoneDataConfig object with data read from the config section.
    """
    columns = [ColumnConfig(x['result'],
                            x['column'],
                            x['shares'].copy() if 'shares' in x else None,
                            _get_collision_handler(x))
                    for x in file_conf['columns']]
    file_path = Path(file_conf['file']['file_name'])
    if not file_path.is_absolute():
        file_path = base_dir / file_path
    return DataFileConfig(data_file = file_path,
                          extra_arguments=file_conf['file']['extra'],
                          aggregation = _get_aggregation(file_conf),
                          columns=columns)


