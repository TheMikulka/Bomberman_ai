import pygame
from States.Entity.state import State

class WalkingState(State):
    def __init__(self, name) -> None:
        super().__init__(name)
        
    def handle_event(self, keys, controls: list) -> str:
        for control in controls:
            if keys[control]:
                return self._name
        return 'Idle'
            