"""Definition of model module base class"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from typing import Any, Dict, Iterable, NamedTuple, Set


@total_ordering
class Phase(Enum):
    GLOBAL = 0
    SCENARIO = 1
    DEMAND_ITERATION = 2
    TIME_PERIOD = 3
    ASSIGNMENT = 4
    
    def __lt__(self, other: Phase):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class CachedValue(NamedTuple):
    value: Any
    scope: Phase = Phase.GLOBAL


class Module(ABC):
        
    name: str
    _provides: Set[str]

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, CachedValue]:
        """Calculates module results

        Args:
            inputs (Dict[str, Any]): Module inputs. Always contains the required inputs
                and might contain the optional inputs.

        Returns:
            Dict[str, CachedValue]: Results as a dictionary of result name and 
                CachedValue containing the actual value and the scope where the value
                is valid
        """
        pass
            
    def __init__(self, 
                 name: str = None,
                 provides: Iterable[str] =[]):
        self.name = name if name else self.__class__.__name__
        self._provides = set(provides)
    
    @property
    def provides(self) -> Set[str]:
        """Returns the variables returned by the process() method

        Returns:
            Set[str]: Variables returned by the process() method
        """
        return self._provides

    @property
    def requires(self) -> Set[str]:
        """Returns the required input parameters of the process() method

        Returns:
            Set[str]: Required input parameters of the process() method
        """
        optional = len(self.process.__defaults__) if self.process.__defaults__ else 0
        return set(self.process\
            .__code__.co_varnames[1:self.process.__code__.co_argcount - optional])

    @property
    def optional(self) -> Set[str]:
        """Returns the optional input parameters of the process() method

        Returns:
            Set[str]: Optional input parameters of the process() method
        """
        optional = len(self.process.__defaults__) if self.process.__defaults__ else 0
        return set(self.process\
            .__code__.co_varnames[self.process.__code__.co_argcount \
                - optional:self.process.__code__.co_argcount])
    
    def __str__(self) -> str:
        return self.name
