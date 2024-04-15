import pygame
from States.Entity.state import State
from pygame.math import Vector2

class WalkingState(State):
    def __init__(self, name) -> None:
        super().__init__(name)
        
    def handle_event(self, direction:Vector2) -> str:
        if direction != Vector2(0, 0):
            return self._name
        return 'Idle'