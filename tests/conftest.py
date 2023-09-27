import logging
from pathlib import Path

import pytest

from model_data.zone_mapping import ZoneMapping

LOGGER = logging.getLogger(__name__)

NETWORK_DATA_FILE = './tests/data/network.gpkg'
KNOWN_ZONES = range(110, 180, 10)

@pytest.fixture(name='mapping', scope='module')
def fixture_mapping() -> ZoneMapping:
    """Returns test ZoneMapping from GTFS file. Reuse previous mapping if available 
    and allow_reuse is True
    
    Args:
        allow_reuse (bool): Allow reuse of previously loaded mapping. Default True.
    Returns:
        ZoneMapping: Test ZoneMapping read from data files
    """
    LOGGER.error('Creating mapping')
    return ZoneMapping.from_gpkg(Path(NETWORK_DATA_FILE))
    