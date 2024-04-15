import pygame
from States.Entity.state import State

class DyingState(State):
    def __init__(self, name) -> None:
        super().__init__(name)
        
    def handle_event(self, keys, controls: list) -> str:
        return self._name