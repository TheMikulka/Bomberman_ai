from Entities.entity import Entity
from map import Map
import pygame
from pygame.math import Vector2
from Obstacles.wall import Wall
from Obstacles.crate import Crate
from bomb import Bomb
from Utilities.settings import *

class AiPlayer(Entity):
    def __init__(self, coords: tuple, entity_name: str, n_frames: tuple, s_width: int, s_height: int, scale, map: Map, game_display: pygame.display) -> None:
        super().__init__(coords[0], coords[1], entity_name, None, n_frames, s_width, s_height, scale, map)
        self._movement_speed = 5
        self._map.add_player(self)
        self.__grid = self._map.current_map
        self.__game_display = game_display
        self.__bombs = pygame.sprite.Group()
        self.__max_bombs = 1
        self.__bomb_strength = 2
        
    def move_left(self) -> None:
        self._direction.x = -1
        if self._current_state != self.states['Walking']:
            self._current_state = self.states['Walking']
            self._current_frame = 0
        self._move_vertical()
        
    def move_right(self) -> None:
        self._direction.x = 1
        if self._current_state != self.states['Walking']:
            self._current_state = self.states['Walking']
            self._current_frame = 0
        self._move_vertical()
    
    def move_up(self) -> None:
        self._direction.y = -1
        if self._current_state != self.states['Walking']:
            self._current_state = self.states['Walking']
            self._current_frame = 0
        self._move_horizontal()
    
    def move_down(self) -> None:
        self._direction.y = 1
        if self._current_state != self.states['Walking']:
            self._current_state = self.states['Walking']
            self._current_frame = 0
        self._move_horizontal()
        
    def stop_moving(self) -> None:
        self._direction = Vector2(0, 0)
        self._current_state = self.states['Idle']
        self._current_frame = 0
        
    def make_move(self, move: str) -> None:
        if self._current_state != self.states['Dying']:
            if move == 'left':
                self.move_left()
            elif move == 'right':
                self.move_right()
            elif move == 'up':
                self.move_up()
            elif move == 'down':
                self.move_down()
            elif move == 'place_bomb':
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
                        bomb.add_observer(self._map)
                        self._map.bombs.add(bomb)
                        self.__bombs.add(bomb)
                        self.__grid[self.__grid.index(row)][row.index(tile)] = bomb
    
    def update(self, delta_time) -> None:
        collidables = [tile.rect for row in self.__grid for tile in row if isinstance(tile, Wall) or isinstance(tile, Crate)]
        if not self._current_state == self.states['Dying']:
            for bomb in self._map.bombs:
                if self.rect.colliderect(bomb.rect):
                    continue
                else:
                    collidables.append(bomb.rect)
            
            self._move_horizontal()
            self.__handle_horizontal_collisions(self._horizontal_collisions(collidables))
            
            self._move_vertical()
            self.__handle_vertical_collisions(self._vertical_collisions(collidables))

            #pygame.draw.rect(game_display, GREEN, (self.rect.x, self.rect.y, self.rect.w, self.rect.h))
            
            super().update(self.__game_display, delta_time)
            
        else:
            for bomb in self._map.bombs:
                if self.rect.colliderect(bomb.rect):
                    continue
                else:
                    collidables.append(bomb.rect)
            
            super().update(self.__game_display, delta_time)

        