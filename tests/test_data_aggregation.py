from pathlib import Path
from typing import cast
import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal

from model_data.zone_mapping import ZoneMapping
from model_data.data_loader import (IndexAggregation, AreaShareAggregation,
                                    LargestAreaAggregation)
from model_data.constants import GEO_ENGINE
import logging

LOGGER = logging.getLogger(__name__)

KNOWN_ZONES = range(110, 180, 10)
VALID_TEST_DATA = pd.DataFrame([{'a': x+1, 'b': x+2} for x in KNOWN_ZONES],
                               index = KNOWN_ZONES)
ZONE_DATA = Path('./tests/data/zone_data.gpkg')
POLYGON_DATA = 'polygon_data'

AREA_SHARE_RESULT = pd.DataFrame(index=pd.Index(KNOWN_ZONES, name='zone_id'),
                                 data=[[37.5         , 39.375],
                                       [83.3333333333, 87.9166666667],
                                       [37.5         , 39.375],
                                       [37.5         , 40.625],
                                       [37.5         , 40.625],
                                       [0.0          , 1.0],
                                       [0.0          , 1.0]],
                                 columns=['a', 'b'])

LARGEST_AREA_RESULT = pd.DataFrame(index=pd.Index(KNOWN_ZONES, name='zone_id'),
                                 data=[[200., 210.],
                                       [200., 210.],
                                       [200., 210.],
                                       [100., 110.],
                                       [100., 110.],
                                       [0.  , 1.],
                                       [0.  , 1.]],
                                 columns=['a', 'b'])

def test_index_aggregation(mapping: ZoneMapping):
    """Test loading zone data from a file. Expect right number of zones to be 
    read"""
    # Check if right number of zones were read
    converted = IndexAggregation()(mapping, VALID_TEST_DATA)
    assert_frame_equal(converted.data, VALID_TEST_DATA)
    
def test_area_share_aggregation(mapping: ZoneMapping):
    """Tests area_share_aggregation
    """
    input_data = cast(gpd.GeoDataFrame, gpd.read_file(ZONE_DATA,
                                                      layer=POLYGON_DATA,
                                                      engine=GEO_ENGINE))
    default_values = {'a': 0., 'b': 1.}
    mapped_data = AreaShareAggregation()(mapping,
                                         input_data,
                                         default_values=default_values)
    assert_frame_equal(mapped_data.data, AREA_SHARE_RESULT)

def test_largest_area_aggregation(mapping: ZoneMapping):
    """Tests largest_area_aggregation
    """
    input_data = cast(gpd.GeoDataFrame, gpd.read_file(ZONE_DATA,
                                                      layer=POLYGON_DATA,
                                                      engine=GEO_ENGINE))
    default_values = {'a': 0., 'b': 1.}
    mapped_data = LargestAreaAggregation()(mapping,
                                         input_data,
                                         default_values=default_values)
    assert_frame_equal(mapped_data.data, LARGEST_AREA_RESULT)
    