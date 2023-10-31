"""Model system module that loads the """
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from model_data.data_item import DataItem, data_items_to_dataframe

from model_data.data_loader import (DataFileConfig, 
                                    section_to_config,
                                    load_files)
from model_data.module import CachedValue, Module
from model_data.zone_mapping import ZoneMapping


class ZoneDataModule(Module):
    config: List[DataFileConfig]
    _explode_data_items: bool
    def __init__(self, base_json: Path, section: str, explode_data_items=True):
        
        with open(base_json) as file:
            self.config = [section_to_config(c) for c in json.load(file)[section]]
        self._explode_data_items = explode_data_items

        results = [c.result 
                     for f in self.config 
                     for c in f.columns]
        if self._explode_data_items:
            results += [s 
                        for f in self.config 
                        for c in f.columns 
                        for s in (c.shares.keys() if c.shares else set()) 
                        if c.shares is not None]

        super().__init__(provides= results)        

    def _get_exploded_view(self, data: pd.DataFrame) -> Dict[str, CachedValue]:
        res = {}
        for col, val in data.items():
            if isinstance(val.iloc[0], DataItem):
                cat_data = data_items_to_dataframe(val)
                res.update(cat_data.to_dict('series'))
            else:
                res[col] = val
        return res
                
    
    def process(self, zone_mapping: ZoneMapping) -> Dict[str, CachedValue]:
        data = load_files(configs=self.config, zone_mapping=zone_mapping)
        if self._explode_data_items:
            res =  self._get_exploded_view(data.data)
        else:
            res = data.data
        return dict([(col_name, CachedValue(col_values)) 
                     for col_name, col_values in res.items()])
            
