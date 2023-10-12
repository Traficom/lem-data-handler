
from concurrent.futures import Executor, Future, wait
from typing import Any, Dict, Sequence, Set
from model_data.module import Module, CachedValue, Phase

class MissingDependencyError(RuntimeError):
    def __init__(self, msg: str):
        super(msg)

class CyclicDependencyError(RuntimeError):
    def __init__(self, msg: str):
        super(msg)

class InvalidModuleError(RuntimeError):
    def __init__(self, msg: str):
        super(msg)



class Environment:

    modules: set[Module]
    providers: Dict[str, Module]
    variables: Dict[str, CachedValue]
    
    def __init__(self):
        self.modules = set()
        self.providers = {}
        self.variables = {}
        
    def __getitem__(self, item: str) -> Any:
        if item in self.variables:
            return self.variables[item].value
        module = self.providers[item]
        self._process_module_results(module,
                                     module.process(**self._get_module_input(module)))
        return self.variables[item].value

    def __setitem__(self, item: str, value: CachedValue) -> Any:
        self.variables[item] = value

    def get(self, item: str, default: Any = None) -> Any:
        return self.variables.get(item, CachedValue(default)).value
    
    def register(self, module: Module):
        provided = module.provides
        collision =  set(self.providers.keys()).intersection(provided)
        if collision:
            items = ', '.join([f'{k}({self.providers[k].name})' 
                               for k in collision])
            msg = f'Data provided by {module.name} collides with existing data: {items}'
            raise KeyError(msg)
        self.providers.update(dict([(k, module) for k in provided]))
        self.modules.add(module)

    def unregister(self, module: Module):
        [self.providers.pop(k) for k in module.provides]
        self.modules.pop(module)

    @property
    def registered_variables(self) -> Sequence[str]:
        return self.providers.keys()
    
    @property
    def cached_variables(self) -> Sequence[str]:
        return self.cached_variables.keys()

    def reset(self, phase: Phase = Phase.GLOBAL):
        """Resets the cached data for a specified model phase

        Args:
            phase (Phase, optional): Model phase to run next. All
                data with scope equal or higher than specified phase will be reset. 
                Defaults to Phase.GLOBAL (resets all data).
        """
        discarded = [self.variables.pop(k) 
                     for k, v in self.variables.items() if v.scope >= phase]
        if discarded:
            self._valid = False

    def _get_module_input(self, module: Module) -> Dict[str, Any]:
        req = [(k, self[k]) for k in module.requires]
        opt = [(k, self[k]) for k in module.optional 
               if k in self.variables]
        return dict(req+opt)
    
    def _process_module_results(self,
                               module: Module,
                               results: Dict[str, CachedValue]):
        missing_results = set(module.provides).difference(results.keys())
        if missing_results:
            msg = (f'{module.name} _process method does not provide all required '
                   f'results: {missing_results}')
            raise InvalidModuleError(msg)
        self.variables.update(results)

    def _ready_to_process(self) -> Set[Module]:
        missing_variables = set(self.providers.keys())\
            .difference(set(self.variables.keys()))
        if not missing_variables:
            return set()
        needs_processing = {self.providers[var] for var in missing_variables}
        result = {module for module in needs_processing 
                if all(dep in self.variables for dep in module.requires)}
        if not result:
            msg = 'Not all modules could be executed due to cyclic dependencies.'
            raise CyclicDependencyError(msg)
        return result

    def process_modules(self,
                        executor: Executor):
        already_processed: Set[Module] = set()
        futures_to_modules: Dict[Future, Module] = {}
        
        def enqueue(modules: Set[Module]):
            new_modules = modules.difference(already_processed)
            already_processed.update(new_modules)
            futures = [executor.submit(module.process,
                                       **self._get_module_input(module)) 
                for module in new_modules]
            futures_to_modules.update(dict(zip(futures, new_modules)))

        enqueue(self._ready_to_process())        
        while futures_to_modules.keys():
            done, _ = wait(futures_to_modules.keys(),
                                  return_when="FIRST_COMPLETED")
            for future in done:
                module = futures_to_modules.pop(future)
                self._process_module_results(module, future.result())
                enqueue(self._ready_to_process())
