import pygame
from Utilities.settings import *

class Tile():
    def __init__(self, x: int, y: int, tile_size: float) -> None:
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.rect = pygame.Rect(self.x, self.y, self.tile_size, self.tile_size)
        
    def update(self, game_display: pygame.display) -> None:
        pass
        