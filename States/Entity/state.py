import pygame

class State():
    def __init__(self, name) -> None:
        self._name = name
        
    def get_name(self) -> str:
        return self._name
    
    def handle_event(self, keys, controls: list) -> str:
        pass