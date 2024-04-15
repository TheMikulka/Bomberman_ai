from abc import ABC, abstractmethod
from Utilities.observable_object import ObservableObject

class Observer(ABC):
    @abstractmethod
    def update(self, observable_object: ObservableObject):
        pass