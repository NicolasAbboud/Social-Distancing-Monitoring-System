# Import necessary libraries
from functools import partial

# Define a Registry class to manage module registrations
class Registry(object):
    
    # Initialize the registry with a name and an empty dictionary to store modules
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
    
    # Return a string representation of the registry, showing its name and registered items
    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str
    
    # Property to get the name of the registry
    @property
    def name(self):
        return self._name
    
    # Property to get the dictionary of registered modules
    @property
    def module_dict(self):
        return self._module_dict
    

    def get(self, key):
        """
        Retrieve a module from the registry by key.
        
        Args:
            key (str): The name of the module to retrieve.
        
        Returns:
            object: The module associated with the given key.
        
        Raises:
            KeyError: If the module is not found in the registry.
        """
        obj = self._module_dict.get(key, None)
        if obj is None:
            raise KeyError(
                f'{key} is not in the {self._name} registry.')
        return obj

    def _register_module(self, module_class, force=False):
        """
        Register a module class in the registry.
        
        Args:
            module_class (type): The module class to register.
            force (bool, optional): Whether to force registration if the module is already registered.
        
        Raises:
            TypeError: If the module is not a class.
            KeyError: If the module is already registered and force is not True.
        """
        if not isinstance(module_class, type):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class


    def register_module(self, cls=None, force=False):
        """
        Decorator to register a module class.
        
        Args:
            cls (type, optional): The module class to register. If None, returns a partial function.
            force (bool, optional): Whether to force registration if the module is already registered.
        
        Returns:
            type: The registered module class.
        """
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls


def build_from_cfg(cfg, registry, **kwargs):
    """
    Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
        
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif isinstance(obj_type, type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    return obj_cls(**args, **kwargs)
