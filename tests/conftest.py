from __future__ import annotations
import logging
from pathlib import Path

import pytest

from model_data.zone_mapping import ZoneMapping
from model_data.data_item import DataItem, dataframe_to_data_items
from pandas import Series, read_csv

LOGGER = logging.getLogger(__name__)

NETWORK_DATA_FILE = './tests/data/network.gpkg'
KNOWN_ZONES = range(110, 180, 10)
TEST_DATA = Path('./tests/data/data.csv')

@pytest.fixture(name='mapping', scope='module')
def fixture_mapping() -> ZoneMapping:
    """Returns test ZoneMapping from GPKG file.
    
    Returns:
        ZoneMapping: Test ZoneMapping read from data files
    """
    LOGGER.error('Creating mapping')
    return ZoneMapping.from_gpkg(Path(NETWORK_DATA_FILE))
    
@pytest.fixture(name='data_items', scope='module')
def fixture_data_items() -> Series[DataItem[int]]:
    """Returns test DataItems for the various test cases.

    Returns:
        Series[DataItem[int]]: DataItems in a pandas Series
    """
    df = read_csv(TEST_DATA, sep='\t', index_col='zone')
    return dataframe_to_data_items(df=df,
                                   total_column='b',
                                   category_columns={1: 's1', 2: 's2'})
