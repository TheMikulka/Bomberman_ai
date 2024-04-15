import pygame
from Utilities.settings import *
from Utilities.sprite_loader import *
from Utilities.observable_object import ObservableObject
from Obstacles.crate import Crate

class Bomb(pygame.sprite.Sprite, ObservableObject):
    def __init__(self, x: int, y: int, bomb_strength: int, game_display: pygame.display) -> None:
        pygame.sprite.Sprite.__init__(self)
        ObservableObject.__init__(self)
        self.x = x
        self.y = y
        self.__game_display = game_display
        width, height = self.__game_display.get_size()
        self.images = get_sprites('Assets/Bomb/bomb.png', 6, (120 * width // width), (120 * height // height), 0.49)
        self.explosion_images = load('Explosions', (12, 12, 12), 120, 120, 0.57)
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
        self.explosion_radius = bomb_strength
        self.__loop_counter = 0
        self.__ticking_speed = 0.1
        self.__explosion_speed = 0.05
        self.current_delta_time = 0
        self.__exploded = False
        self.directions = {}
    
    def notify_observers(self) -> None:
        for observer in self._observers:
            observer.update(self)
    
    def __explode(self) -> None:
        self.notify_observers()
        
    def set_explosion(self, explosion_tiles: dict) -> None:
        self.__exploded = True
        self.directions = explosion_tiles
        
    def __check_direction(self, direction: str, image: pygame.Surface) -> pygame.Surface:
        match direction:
            case 'up':
                return pygame.transform.rotate(image, -90).convert_alpha()
            case 'down':
                return pygame.transform.rotate(image, 90).convert_alpha()
            case 'right':
                return pygame.transform.flip(image, True, False).convert_alpha()
            case 'left':
                return image
            
        
    def update(self, delta_time) -> None:
        self.current_delta_time += delta_time
        if not self.__exploded:
            if self.current_delta_time >= self.__ticking_speed:
                if self.index < len(self.images) - 1:
                    self.index += 1

                if self.index >= len(self.images) - 1:
                    self.index = 0
                    self.__loop_counter += 1

                if self.__loop_counter >= 3:
                    self.__explode()

                self.current_delta_time = 0
                
            current_image = self.images[self.index]
            self.__game_display.blit(current_image, (self.x, self.y))
        else:
            if self.current_delta_time >= self.__explosion_speed:
                if self.index < len(self.explosion_images['middle']) - 1:
                    self.index += 1
                else:
                    self.kill()
                    
                self.current_delta_time = 0
                
            for key in self.directions:
                for i in range(len(self.directions[key])):
                    tile, x, y = self.directions[key][i]
                    if key == 'center':
                        self.__game_display.blit(self.explosion_images['center'][self.index], (self.rect.x, self.rect.y))
                    elif i == len(self.directions[key]) - 1:
                        self.__game_display.blit(self.__check_direction(key, self.explosion_images['end'][self.index]), (tile.rect.x, tile.rect.y))
                        break
                    elif not isinstance(tile, Crate):
                        self.__game_display.blit(self.__check_direction(key, self.explosion_images['middle'][self.index]), (tile.rect.x, tile.rect.y))

            
                
                    
        