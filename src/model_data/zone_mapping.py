"""Definition of zoning and mapping between zone ids and offsets"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, cast

import geopandas as gpd
import openmatrix as omx
import pandas as pd
from shapely.geometry import Point, Polygon

from model_data.constants import GEO_ENGINE

SELECTED_ZONE_DATA = [
    'centroid',
    'polygon'
]

CENTROID_ID_COLUMN = 'id'
POLYGON_ID_COLUMN = 'zone_id'

class IncompatibleZoneException(Exception):
    """Raised when trying to combine incompatible zone mappings."""
    def __init__(self, message: str):
        super().__init__(message)
    

try:
    import pandera as pa
    from pandera.typing import DataFrame, Index
    from pandera.typing.geopandas import GeoSeries
    USE_DATA_VALIDATION = True
    class ZoneDataSchema(pa.DataFrameModel):
        """Schema definition for zone data"""
        zone_id: Index[pa.Int64] = pa.Field(ge=0, check_name=True)
        centroid: GeoSeries[Point] = pa.Field(nullable=True)
        polygon: GeoSeries[Polygon] = pa.Field(nullable=True)
    DataFrameType = DataFrame[ZoneDataSchema]
except ImportError:
    from geopandas import GeoSeries
    from pandas import DataFrame, Index
    USE_DATA_VALIDATION = False
    DataFrameType = DataFrame

class Zone():
    """Represenstation of a single traffic assignemetn zone"""
    _mapping: ZoneMapping
    """Reference to the zone mapping used by the zone""" 
    _offset: int
    """Offset of the zone"""

    def __init__(self, mapping: ZoneMapping, offset: int):
        """Initializes the zone wrapper

        Args:
            mapping (ZoneMapping): Reference to the zone mapping to use
            offset (int): Offset of the zone
        """
        self._mapping = mapping
        self._offset = offset

    @property
    def offset(self) -> int:
        """Matrix offset of the zone"""
        return self._offset

    @property
    def zone_id(self) -> int:
        """Zone ID"""
        return self._mapping.zone_data.index[self._offset]

    @property
    def centroid(self) -> Point:
        """Centroid of the zone"""
        return self._mapping.zone_data.iloc[self._offset].centroid

    @property
    def polygon(self) -> Polygon:
        """Borders of the zone"""
        return self._mapping.zone_data.iloc[self._offset].polygon

class ZoneMapping:
    """Zone mapping of zone IDs and offsets"""

    zone_data: DataFrameType

    @classmethod
    def from_gpkg(cls, filename: Path) -> ZoneMapping:
        """Creates new zone mapping from network GPKG file.

        Args:
            filename (Path): Path to the GPKG file containing the network data

        Returns:
            ZoneMapping: Zone mapping object with content read from the specified file
        """
        centroids = cast(pd.DataFrame, gpd.read_file(filename=filename,
                            layer='centroids',
                            engine=GEO_ENGINE)[[CENTROID_ID_COLUMN, 'geometry']])\
                .rename(columns={'geometry': 'centroid'})\
                .set_index(CENTROID_ID_COLUMN)
        centroids.index = centroids.index.astype('int64')
        
        polygons = cast(pd.DataFrame, gpd.read_file(filename,
                            layer='centroid_polygons',
                            engine=GEO_ENGINE)[[POLYGON_ID_COLUMN, 'geometry']])\
            .rename(columns={'geometry': 'polygon'}) \
            .set_index(POLYGON_ID_COLUMN)
        polygons.index = polygons.index.astype('int64')
        result = DataFrameType(polygons.join(centroids))
        return ZoneMapping(result)

    @classmethod
    def from_zone_numbers(cls, zones: Iterable[int]):
        data = pd.DataFrame(data={
                                'centroid': gpd.GeoSeries(),
                                'polygon': gpd.GeoSeries(),
                            },
                            index=pd.Index(data=zones, 
                                           name='zone_id', 
                                           dtype='int64'))
        result = DataFrameType(data)
        return ZoneMapping(result)

    @classmethod
    def from_omx(cls, filename: Path) -> ZoneMapping:
        """Creates new zone mapping from OMX file.

        Args:
            filename (Path): Path to the OMX file containing the zone mapping

        Returns:
            ZoneMapping: Zone mapping object with content read from the specified file
        """
        file = omx.open_file(filename)
        zones_in_file = file.map_entries('zone_number')
        file.close()
        return ZoneMapping.from_zone_numbers(zones_in_file)

    @classmethod
    def from_file(cls, filename: Path) -> ZoneMapping:
        """Creates zonemapping from a file and tries to guess the right format using
            the file extension

        Args:
            filename (Path): Path to the file to load

        Returns:
            ZoneMapping: New zone mapping read from the file
        """
        loaders = {
            '.gpkg': ZoneMapping.from_gpkg,
            '.omx': ZoneMapping.from_omx,
        }
        extension = filename.suffix
        return loaders[extension](filename)

    def __init__(self, data: DataFrameType):
        """Initializes zone mapping from given geodataframe.
        Args:
            data (DataFrameType): The dataframe must contain the following 
                index:
                    zone_id (int): Unique ID number of the zone.
                columns:
                    centroid (shapely.geometry.Point): Point geometry for the centroid
                    polygon (shapely.geometry.Polygon): Zone borders as polygon geometry
        """
        selected_data = data[SELECTED_ZONE_DATA].sort_index()
       
        self.zone_data = cast(DataFrameType, selected_data)
        self.zone_data.index.name = POLYGON_ID_COLUMN

    def has_geometry(self) -> bool:
        """Check if the mapping has valid geometry definitions

        Returns:
            bool: True if polygon geometries are defined and have non zero areas
        """
        areas = gpd.GeoSeries(self.zone_data.polygon).area
        return not (areas.isnull().any() or areas.isin([None, 0.0]).any())
        
    def get_zone_by_id(self, zone_id: int) -> Zone:
        """Returns a zone with the specified id

        Args:
            zone_id (int): Zone id of the requested zone

        Returns:
            Zone: Zone object with the given zone id
        """
        offset = cast(int, self.zone_data.index.get_loc(zone_id))
        return self.get_zone_by_offset(offset)

    def get_zone_by_offset(self, offset: int) -> Zone:
        """Returns zone with the specified offset

        Args:
            offset (int): Offset (0...n_zones) of the requested zone.

        Returns:
            Zone: Zone object with the given offset
        """
        return Zone(self, offset)

    def offsets_to_zone_ids(self, offsets: Iterable[int]) -> List[int]:
        """Convers a sequence of offsets into list of zone_ids

        Args:
            offsets (Iterable[int]): Sequence of zone offsets

        Returns:
            List[int]: List of matching zone IDs
        """
        values = self.zone_data.iloc[list(offsets)].index
        return list(values)

    def zone_ids_to_offsets(self, zone_ids: Iterable[int]) -> List[int]:
        """Converts a sequence of sone IDs into list on offsets

        Args:
            ids (Iterable[int]): Sequence of zone IDs

        Returns:
            List[int]: List of matching zone offsets
        """
        return [self.zone_data.index.get_loc(i) for i in zone_ids]
    
    def __len__(self) -> int:
        """Returns the number of zones in the zonemapping.

        Returns:
            int: Number of zones speficied in the mapping"""
        return self.zone_data.shape[0]

    def __iter__(self) -> ZoneIterator:
        """Iterates all the zones in the mapping

        Returns:
            Iterable[Zone]: _description_

        Yields:
            Iterator[Iterable[Zone]]: _description_
        """
        return ZoneIterator(self)

    @property
    def centroids(self) -> GeoSeries[Point]:
        return GeoSeries(self.zone_data['centroid'])

    @property
    def polygons(self) -> GeoSeries[Polygon]:
        return GeoSeries(self.zone_data['polygon'])
    
    @property
    def zones(self) -> List[int]:
        """The zone IDs as a list

        Returns:
            List[int]: The zone IDs as a list
        """
        return self.zone_data.index.to_list()

class ZoneIterator:
    """Iterator for ZoneMapping
    """
    _mapping: ZoneMapping
    _i: int
    _max: int

    def __init__(self, mapping: ZoneMapping):
        self._mapping = mapping
        self._i = 0
        self._max = len(self._mapping)

    def __iter__(self) -> ZoneIterator:
        return self

    def __next__(self) -> Zone:
        if self._i >= self._max:
            raise StopIteration
        self._i += 1
        return Zone(self._mapping, self._i - 1)
