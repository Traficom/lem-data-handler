"""Model system module that loads the """
import json
from pathlib import Path
from typing import Dict
from model_data.module import Module, CachedValue, Phase
from model_data.zone_mapping import ZoneMapping

class ZoneMappingModule(Module):
    zone_file: Path
    def __init__(self, base_json: Path):
        
        with open(base_json) as file:
            config = json.load(file)
        self.zone_file = config['zone_mapping']['zone_file']
        super().__init__(provides=['zone_mapping'])
    
    def process(self) -> Dict[str, CachedValue]:
        return {'zone_mapping': 
                    CachedValue(ZoneMapping.from_file(Path(self.zone_file)),
                                Phase.SCENARIO)}