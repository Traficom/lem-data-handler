from __future__ import annotations
import logging

from pytest import approx
from pandas import Series
from model_data.data_item import DataItem

LOGGER = logging.getLogger(__name__)

def test_data_item_loading(data_items: Series[DataItem[int]]):
    """Test loading data items from csv file"""
    data_130 = data_items.loc[130]
    assert data_130.total == approx(5.0), "Total does not match"
    assert data_130[2] == approx(5.0*0.4), "S2 does not match"
    
def test_data_item_addition(data_items: Series[DataItem[int]]):
    """Test adding two DataItems"""
    data_sum = data_items.loc[110] + data_items.loc[130]
    assert data_sum.total == approx(2.0 + 5.0), "Total does not match"
    assert data_sum[2] == approx(2.0*0.2 + 5.0*0.4), "S2 does not match"
    
def test_data_item_multiply(data_items: Series[DataItem[int]]):
    """Test multiplying DataItem"""

    data_130 = data_items.loc[130] * 2.0
    assert data_130.total == approx(5.0 * 2.0), "Total does not match"
    assert data_130[2] == approx(5.0*0.4 * 2.0), "S2 does not match"
    
def test_data_item_divide(data_items: Series[DataItem[int]]):
    """Test dividing DataItem"""

    data_130 = data_items.loc[130] / 2.0
    assert data_130.total == approx(5.0 / 2.0), "Total does not match"
    assert data_130[2] == approx(5.0*0.4 / 2.0), "S2 does not match"