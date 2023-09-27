"""Unittests for zone mapping"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from model_data.zone_mapping import ZoneMapping
from model_data.zone_transformer import AreaShareTransformer
from numpy.testing import assert_almost_equal

LOGGER = logging.getLogger(__name__)

NUMBER_OF_AREAS_A = 7
NUMBER_OF_AREAS_B = 6
NETWORK_DATA_FILE1 = './tests/data/network.gpkg'
NETWORK_DATA_FILE2 = './tests/data/network2.gpkg'
TEST_INPUT_VECTOR_A = np.array(range(1, NUMBER_OF_AREAS_A+1))
TEST_OUTPUT_VECTOR_A = np.array([
    TEST_INPUT_VECTOR_A[0]*1.0,
    TEST_INPUT_VECTOR_A[1]*0.5,
    TEST_INPUT_VECTOR_A[1]*0.5,
    TEST_INPUT_VECTOR_A[2]*0.5,
    TEST_INPUT_VECTOR_A[3]*0.5 + TEST_INPUT_VECTOR_A[4]*0.5,
    0.0
])

TEST_MATRIX_A = np.array(range(1, NUMBER_OF_AREAS_A**2 + 1))\
    .reshape((NUMBER_OF_AREAS_A, NUMBER_OF_AREAS_A))

TEST_MATRIX_B = np.zeros((NUMBER_OF_AREAS_B, NUMBER_OF_AREAS_B))
TEST_MATRIX_B[0,0] = TEST_MATRIX_A[0,0]
TEST_MATRIX_B[0,1] = TEST_MATRIX_A[0,1] * 0.5
TEST_MATRIX_B[0,2] = TEST_MATRIX_A[0,1] * 0.5
TEST_MATRIX_B[0,3] = TEST_MATRIX_A[0,2] * 0.5
TEST_MATRIX_B[0,4] = (TEST_MATRIX_A[0,3] + TEST_MATRIX_A[0,4]) * 0.5
TEST_MATRIX_B[0,5] = 0

TEST_MATRIX_B[1,0] = TEST_MATRIX_A[1,0] * 0.5
TEST_MATRIX_B[1,1] = TEST_MATRIX_A[1,1] * 0.5 * 0.5
TEST_MATRIX_B[1,2] = TEST_MATRIX_A[1,1] * 0.5 * 0.5
TEST_MATRIX_B[1,3] = TEST_MATRIX_A[1,2] * 0.5 * 0.5
TEST_MATRIX_B[1,4] = 0.5 * (TEST_MATRIX_A[1,3] * 0.5 + TEST_MATRIX_A[1,4] * 0.5)
TEST_MATRIX_B[1,5] = 0

TEST_MATRIX_B[2,:] = TEST_MATRIX_B[1,:]

TEST_MATRIX_B[3,0] = TEST_MATRIX_A[2,0] * 0.5
TEST_MATRIX_B[3,1] = TEST_MATRIX_A[2,1] * 0.5 * 0.5
TEST_MATRIX_B[3,2] = TEST_MATRIX_A[2,1] * 0.5 * 0.5
TEST_MATRIX_B[3,3] = TEST_MATRIX_A[2,2] * 0.5 * 0.5
TEST_MATRIX_B[3,4] = 0.5 * (TEST_MATRIX_A[2,3] * 0.5 + TEST_MATRIX_A[2,4] * 0.5)
TEST_MATRIX_B[3,5] = 0

TEST_MATRIX_B[4,0] = TEST_MATRIX_A[3, 0] * 0.5 + TEST_MATRIX_A[4, 0] * 0.5
TEST_MATRIX_B[4,1] = 0.5*(TEST_MATRIX_A[3,1] * 0.5 + TEST_MATRIX_A[4,1] * 0.5)
TEST_MATRIX_B[4,2] = 0.5*(TEST_MATRIX_A[3,1] * 0.5 + TEST_MATRIX_A[4,1] * 0.5)
TEST_MATRIX_B[4,3] = 0.5*(TEST_MATRIX_A[3,2] * 0.5 + TEST_MATRIX_A[4,2] * 0.5)
TEST_MATRIX_B[4,4] = 0.5*(TEST_MATRIX_A[3,3] * 0.5 + TEST_MATRIX_A[3,4] * 0.5) \
    + 0.5*(TEST_MATRIX_A[4,3] * 0.5 + TEST_MATRIX_A[4,4] * 0.5)
TEST_MATRIX_B[4,5] = 0

TEST_MATRIX_B[5,:] = 0

@pytest.fixture(name='mappings', scope='module')
def fixture_mappings() -> Tuple[ZoneMapping, ZoneMapping]:
    """Returns test ZoneMapping from GTFS file. Reuse previous mapping if available and
    allow_reuse is True
    
    Args:
        allow_reuse (bool): Allow reuse of previously loaded mapping. Default True.
    Returns:
        Tuple[ZoneMapping, ZoneMapping]: Tuple of two ZoneMappings read from data files
    """
    maps = (
        ZoneMapping.from_gpkg(Path(NETWORK_DATA_FILE1)),
        ZoneMapping.from_gpkg(Path(NETWORK_DATA_FILE2))
    )
    return maps

def test_area_share_transformer_matrix(mappings: Tuple[ZoneMapping, ZoneMapping]):
    """Test loading zone data from a file. Expect right number of zones to be read"""
    mapping1, mapping2 = mappings
    transformer = AreaShareTransformer(mapping1, mapping2)
    result = transformer.transform_matrix(TEST_MATRIX_A ).round(2)
    assert_almost_equal(result, TEST_MATRIX_B)
    
def test_area_share_transformer_vectors_single(mappings: Tuple[ZoneMapping,
                                                               ZoneMapping]):
    """Test loading zone data from a file. Expect right number of zones to be read"""
    mapping1, mapping2 = mappings
    transformer = AreaShareTransformer(mapping1, mapping2)

    result = transformer.transform_vectors(TEST_INPUT_VECTOR_A)
    assert_almost_equal(result, TEST_OUTPUT_VECTOR_A)

def test_area_share_transformer_vectors_multiple(mappings: Tuple[ZoneMapping,
                                                                ZoneMapping]):
    """Test loading zone data from a file. Expect right number of zones to be read"""
    mapping1, mapping2 = mappings
    transformer = AreaShareTransformer(mapping1, mapping2)

    vector_range = range(1,3)

    vectors = np.stack([TEST_INPUT_VECTOR_A * i for i in vector_range])
    result = transformer.transform_vectors(vectors)
    for i, val in enumerate(vector_range):
        assert_almost_equal(result[i], TEST_OUTPUT_VECTOR_A*val)
