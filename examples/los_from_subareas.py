"""Loads Level-of-Service data from individual subarea matrices"""



from pathlib import Path
from typing import Dict, List

import numpy as np

from model_data.los import (LOS_MAX, LOS_MISSING, LOSData, LOSDtype, LOSMatrix,
                            LOSTimePeriod, LOSType, TimePeriod,
                            get_helmet_matrix_spec)
from model_data.zone_mapping import ZoneMapping

BASE_PATH = Path('../../../SharePoint/T')

DATA_PATH = BASE_PATH / '8. Hankkeessa tuotetut lähtötiedot malleihin/Vastusmatriisit'
SUBAREA_PATHS = [
    DATA_PATH / 'Itä-Suomi',
    DATA_PATH / 'Lounais-Suomi',
    DATA_PATH / 'Pohjois-Suomi',
    DATA_PATH / 'Uusimaa',
]

TARGET_MAPPING_GPKG = BASE_PATH / ('3. Liikenneverkkotyö/Emmeaineistot/vanhat verkot'+
    '/2023-08-10_koko_suomi/koko_suomi.gpkg')
TARGET_PATH= Path('../data/koko_suomi')

def load_subareas(paths: List[Path]) -> List[LOSData]:
    """Loads subarea LOS data using helmet naming scheme

    Args:
        paths (List[Path]): List of paths to the subarea directories

    Returns:
        List[LOSData]: List of LOS data for each subarea
    """
    return [LOSData.from_omx_files(mapping=None, 
                                   omx_specs=get_helmet_matrix_spec(path))
            for path in paths]
    

def gen_test_data():
    mapping = ZoneMapping.from_gpkg(Path('./tests/data/network.gpkg'))
    time_periods: Dict[TimePeriod, LOSTimePeriod] = {}
    offset = 0
    matrix_size = len(mapping)**2
    for time_period in TimePeriod:
        types: Dict[LOSType, LOSMatrix] = {}
        for type in LOSType:
            
            matrix = np.array(range(matrix_size))\
                .reshape((len(mapping), len(mapping))) + offset
            offset += matrix_size
            types[type] = LOSMatrix(mapping, matrix.astype(LOSDtype))
        time_periods[time_period] = LOSTimePeriod(types)
    los = LOSData(time_periods)
    target_path = Path('./tests/data/los')
    los.to_omx_files(get_helmet_matrix_spec(target_path))


def main():
    #gen_test_data()
    # Generate zone mapping for target national model
    national_mapping = ZoneMapping.from_gpkg(TARGET_MAPPING_GPKG)

    # Get subarea LOS data
    subareas = load_subareas(SUBAREA_PATHS)

    # Generate LOS data from subareas using LOG_MISSING as default value for
    # zone pairs not defined in the subareas
    national_los = LOSData.from_subareas(national_mapping, subareas, LOS_MAX)

    # Write national LOS data into OMX files using Helmet naming scheme
    national_los.to_omx_files(get_helmet_matrix_spec(TARGET_PATH))
    #national_los.to_pickle(TARGET_PATH / 'national.pkl')

    #national_los_from_pickle = LOSData.from_pickle(TARGET_PATH / 'national.pkl')
    #national_los_from_omx = LOSData.from_omx_files(get_helmet_matrix_spec(TARGET_PATH),
    #                                             mapping=None)
    print('done')
    # Write national LOS data into OMX files using Helmet naming scheme
    #national_los.to_omx_files(get_helmet_matrix_spec(TARGET_PATH))
    #national_los.to_pickle(TARGET_PATH / 'national.pkl')

    #national_los_from_pickle = LOSData.from_pickle(TARGET_PATH / 'national.pkl')
    #national_los_from_omx = LOSData.from_omx_files(get_helmet_matrix_spec(TARGET_PATH),
    #                                             mapping=None)
    print('done')    #national_los.to_omx_files(get_helmet_matrix_spec(TARGET_PATH))
    #national_los.to_pickle(TARGET_PATH / 'national.pkl')

    #national_los_from_pickle = LOSData.from_pickle(TARGET_PATH / 'national.pkl')
    #national_los_from_omx = LOSData.from_omx_files(get_helmet_matrix_spec(TARGET_PATH),
    #                                             mapping=None)
    print('done')

if __name__ == '__main__':
    main()