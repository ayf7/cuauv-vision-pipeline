from vision.core.base import ModuleBase
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict
import time

"""
A handler is essentially a "hook" for custom processing in vision. It is
intended to serve similarly to a vision module, with possible preprocessing
information.

The current use case for handlers is YOLO. The current vision YOLO vision module
in [yolo.py] is a wrapper for ultralytics' YOLO class, which returns a list of
ultralytics.Results object. This results object is converted into a dictionary
via .summary() method.
"""

class HandlerBase(ABC):
    
    def __init__(
        self,
        name:str,
        parent: ModuleBase = None
    ):
        """
        Initializes a handler for a module. Note that if [parent] is None, is it
        pretty much unusable. Requires calling [register] first before all else.

        Args:
            name:   the identifier of the handler base. In the vision module (i.e.
                    the inherited ModuleBase class), you can retrieve a HanderBase
                    object through self.handlers[<name>].
            parent: the original vision module class. Note that this field will
                    probably be None upon initialization; post-initialization, the
                    register() function will be invoked by the HandlerMixin.


        """
        self._name = name
        self._parent = parent
        if parent is not None:
            self._initialize_methods()
    
    def register(self, parent: ModuleBase):
        self._parent = parent
        self._initialize_methods()
        
    def _initialize_methods(self):
        """
        To create the illusion that the Handler is essentially a vision module,
        we assign all of its functionality to this class.
        """
        self.normalize_axis = self._parent.normalize_axis
        self.normalize = self._parent.normalize
        self.post = self._parent.post
        self.tuners = self._parent.tuners
        self.get_latency = self._parent.get_latency
        self._loop = self._parent._loop
    
    @abstractmethod
    def process(self, direction: str, image: np.ndarray, *args, **kwargs):
        """
        Process step, similar to how ModuleBase process works. It is entirely
        up to the user to define how the parent ModuleBase's function should call
        this function, and what additional parameters should be included in the
        method funciton call.

        Raises:
            NotImplementedError: raised if method not overridden
        """
        raise NotImplementedError("HandlerBase.process")
    
    @property
    def name(self):
        return self._name

class HandlerMixin:
    
    def __init__(
            self,
            handlers: List[HandlerBase]= []
        ):
        self._handlers : Dict[str, HandlerBase] = {} # maps name of handler -> handler itself
        self._handler_names = set()
        for handler in handlers:
            self._handlers[handler.name] = handler
            if handler.name in self._handler_names:
                raise KeyError("Duplicate handler names found!")
            self._handler_names.add(handler.name)
        
        for _, handler in self._handlers.items():
            handler.register(self) # points handler to parent module
    
    @property
    def handlers(self):
        return self._handlers

    @property
    def handler_names(self):
        return self._handler_names