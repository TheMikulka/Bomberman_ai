import pygame
from pygame.math import Vector2
from Entities.entity import Entity
from Obstacles.wall import Wall
from Obstacles.crate import Crate
from bomb import Bomb
from Utilities.settings import *
from Utilities.observable_object import ObservableObject
from map import Map

class Player(Entity):
    def __init__(self, coords: tuple, entity_name: str, key_bindings: list, n_frames: tuple, s_width: int, s_height: int, scale, map: Map, game_display: pygame.display) -> None:
        super().__init__(coords[0], coords[1], entity_name, key_bindings, n_frames, s_width, s_height, scale, map)
        self._movement_speed = 5
        self.__map = map
        self.__map.add_player(self)
        self.__grid = self.__map.current_map
        self.__game_display = game_display
        self.__bombs = pygame.sprite.Group()
        self.__max_bombs = 1
        self.__bomb_strength = 2
        
    def check_keys(self, pressed_keys) -> None:
        self._wanted_direction = Vector2(0, 0)
        
        if (pressed_keys[self._controls[0]]):
            self._move_up()
        elif (pressed_keys[self._controls[1]]):
            self._move_down()
        elif (pressed_keys[self._controls[2]]):
            self._move_left()
        elif (pressed_keys[self._controls[3]]):
            self._move_right()
                
        if pressed_keys[self._controls[4]] and len(self.__bombs) < self.__max_bombs:
            self.place_bomb()
            
    def __handle_horizontal_collisions(self, collided: bool) -> None:
        if collided: self._direction.x = 0
        
    def __handle_vertical_collisions(self, collided: bool) -> None:
        if collided: self._direction.y = 0
    
    
    def place_bomb(self) -> None:
        if len(self.__bombs) < self.__max_bombs:
            for row in self.__grid:
                for tile in row:
                    if self.check_position(tile) and not isinstance(tile, Bomb):
                        bomb = Bomb(tile.rect.x, tile.rect.y, self.__bomb_strength, self.__game_display)
                        bomb.add_observer(self.__map)
                        self.__map.bombs.add(bomb)
                        self.__bombs.add(bomb)
                        self.__grid[self.__grid.index(row)][row.index(tile)] = bomb
        
    def update(self, pressed_keys, delta_time) -> None:
        collidables = [tile.rect for row in self.__grid for tile in row if isinstance(tile, Wall) or isinstance(tile, Crate)]
        if not self._current_state == self.states['Dying']:
            self.check_keys(pressed_keys)
            for bomb in self.__map.bombs:
                if self.rect.colliderect(bomb.rect):
                    continue
                else:
                    collidables.append(bomb.rect)
            
            self._move_horizontal()
            self.__handle_horizontal_collisions(self._horizontal_collisions(collidables))
            
            self._move_vertical()
            self.__handle_vertical_collisions(self._vertical_collisions(collidables))
            
            super().update(self.__game_display, delta_time)
            
        else:
            for bomb in self.__map.bombs:
                if self.rect.colliderect(bomb.rect):
                    continue
                else:
                    collidables.append(bomb.rect)
            
            super().update(self.__game_display, delta_time)