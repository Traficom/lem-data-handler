"""Unittests for zone mapping"""
from model_data.zone_mapping import ZoneMapping
import logging

LOGGER = logging.getLogger(__name__)

"""Unittest class to test tours.person"""

KNOWN_ZONES = range(110, 180, 10)

def test_0_read_from_gpkg(mapping: ZoneMapping):
    """Test loading zone data from a file. Expect right number of zones to be 
    read"""
    # Check if right number of zones were read
    assert mapping.zone_data.shape[0] == len(KNOWN_ZONES)

def test_get_zone_by_id(mapping: ZoneMapping):
    """Test get_zone_by_id method. Expect returned offset to match test data"""
    offsets, zone_ids = zip(*enumerate(KNOWN_ZONES))
    res = [mapping.get_zone_by_id(z).offset for z in zone_ids]
    assert res == list(offsets)

def test_get_zone_by_offset(mapping: ZoneMapping):
    """Test get_zone_by_id method. Expect returned offset to match test data"""
    offsets, zone_ids = zip(*enumerate(KNOWN_ZONES))
    res = [mapping.get_zone_by_offset(i).zone_id for i in offsets]
    assert res == list(zone_ids)

def test_len_operator(mapping: ZoneMapping):
    """Test the len-operator. Expect returned value to match the number of 
    KNOWN_ZONES"""
    assert len(mapping) == len(KNOWN_ZONES)

def test_offsets_to_zone_ids(mapping: ZoneMapping):
    """Test getting multiple zone_ids from offsets"""
    offsets, zone_ids = zip(*enumerate(KNOWN_ZONES))
    res = mapping.offsets_to_zone_ids(offsets)
    assert res == list(zone_ids)

def test_zone_ids_to_offsets(mapping: ZoneMapping):
    """Test getting multiple offsets from zone_ids"""
    offsets, zone_ids = zip(*enumerate(KNOWN_ZONES))
    res = mapping.zone_ids_to_offsets(zone_ids)
    assert res == list(offsets)

def test_zone_mapping_iterator(mapping: ZoneMapping):
    """Tests iteration of zone mapping"""
    res = [(z.offset, z.zone_id) for z in mapping]
    assert res == list(enumerate(KNOWN_ZONES))

def test_zone_mapping_has_geometry_true(mapping: ZoneMapping):
    """Tests has_geometry when the mapping has geometry"""
    res = mapping.has_geometry()
    assert res

def test_zone_mapping_has_geometry_false(mapping: ZoneMapping):
    """Tests has_geometry when the mapping has no geometry"""
    map2 = ZoneMapping.from_zone_numbers(mapping.zones)
    res = map2.has_geometry()
    assert not res
