import pygame

class AssetLoader():
    __instance = None
    
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(AssetLoader, cls).__new__(cls)
        return cls.__instance
    
    def __init__(self) -> None:
        if isinstance(self.__instance, AssetLoader):
            self.wall_img = pygame.image.load('Assets/Wall/wall.png').convert_alpha()
            self.crate_img = pygame.image.load('Assets/Crate/crate.png').convert_alpha()