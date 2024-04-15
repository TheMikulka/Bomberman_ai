import pygame
from pygame.math import Vector2

class State():
    def __init__(self, name) -> None:
        self._name = name
        
    def get_name(self) -> str:
        return self._name
    
    def handle_event(self, direction:Vector2) -> str:
        pass