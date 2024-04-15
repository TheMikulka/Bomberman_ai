from abc import ABC, abstractmethod

class ObservableObject (ABC):
    def __init__(self) -> None:
        self._observers = []
    
    def add_observer(self, observer) -> None:
        self._observers.append(observer)

    def remove_observer(self, observer) -> None:
        self._observers.remove(observer)
    
    @abstractmethod
    def notify_observers(self) -> None:
        pass