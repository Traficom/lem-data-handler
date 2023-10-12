from __future__ import annotations
import logging
from typing import Dict

from model_data.environment import Environment
from model_data.module import Module, CachedValue, Phase
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

LOGGER = logging.getLogger(__name__)

class TestModule1(Module):
    def __init__(self):
        super().__init__(name='test1', provides=['t1_const1', 't1_const2'])
    
    def process(self, ) -> Dict[str, CachedValue]:
        return {
            't1_const1': CachedValue(10),
            't1_const2': CachedValue(True)
        }

class TestModule2(Module):
    def __init__(self):
        super().__init__(name='test2', provides=['t2_const'])
    
    def process(self, t1_const1: int) -> Dict[str, CachedValue]:
        return {
            't2_const': CachedValue(t1_const1 + 1)
        }

class TestModule3(Module):
    def __init__(self):
        super().__init__(name='test3', provides=['t3_sum', 't3_cond'])
    
    def process(self,
                t1_const1: int,
                t2_const: int,
                t1_const2: bool=False,
                non_existing: int=99) -> Dict[str, CachedValue]:
        return {
            't3_sum': CachedValue(t1_const1+t2_const, scope=Phase.GLOBAL),
            't3_cond': CachedValue(t1_const1*non_existing if t1_const2 
                                   else t1_const1+non_existing,
                                   scope=Phase.DEMAND_ITERATION)
        }

def test_module_registeration():
    env = Environment()
    env.register(TestModule1())
    env.register(TestModule2())
    env.register(TestModule3())
    
    assert len(env.registered_variables) == 2+1+2, \
        'The number of provided variables does not match'
    
def test_recursive_processing():
    env = Environment()
    env.register(TestModule1())
    env.register(TestModule2())
    env.register(TestModule3())

    assert env['t3_sum'] == 10+(10+1), 'Varible t3_sum incorrect'
    
def test_optional_parameters():
    env = Environment()
    env.register(TestModule1())
    env.register(TestModule2())
    env.register(TestModule3())

    assert env['t3_cond'] == 10*99, 'Varible t3_cond incorrect'

def test_process_pool_processing():
    env = Environment()
    env.register(TestModule1())
    env.register(TestModule2())
    env.register(TestModule3())

    with ProcessPoolExecutor() as executor:
        env.process_modules(executor)
    assert env.get('t3_cond', None) == 10*99, \
        'Varible t3_sum incorrect after processing with ProcessPoolExecutor'

def test_thread_pool_processing():
    env = Environment()
    env.register(TestModule1())
    env.register(TestModule2())
    env.register(TestModule3())

    with ThreadPoolExecutor() as executor:
        env.process_modules(executor)
    assert env.get('t3_cond', None) == 10*99, \
        'Varible t3_sum incorrect after processing with ThreadPoolExecutor'
