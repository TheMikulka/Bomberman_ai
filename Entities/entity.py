import pygame
from Utilities.sprite_loader import load
from pygame.math import Vector2
from States.Entity.idling_state import IdlingState
from States.Entity.walking_state import WalkingState
from States.Entity.dying_state import DyingState
from Obstacles.crate import Crate
from Obstacles.wall import Wall
from Utilities.settings import *
from map import Map


class Entity():
    def __init__(self, x: int, y: int, entity_name: str, key_bindings: list, n_frames: tuple, s_width: int, s_height: int, scale, map: Map) -> None:
        self._map = map
        self.x = x
        self.y = y
        self._all_actions = load(entity_name, n_frames, s_width, s_height, scale)
        self._controls = key_bindings
        self._direction = Vector2(0, 0)
        self._wanted_direction = Vector2(0, 0)
        self._current_delta_time = 0
        self._animation_speed = 0.1
        self._current_frame = 0
        self._current_state = IdlingState('Idle')
        
        
        self.states = {'Idle': IdlingState('Idle'), 'Walking': WalkingState('Walking'), 'Dying': DyingState('Dying')}
        
        self.image = self._all_actions['Idle']['front'][0]
        
        self.__shrink_width = 30
        self.__shrink_height = 30
        
        self.rect = self.image.get_rect(topleft=(x, y))
        self.rect.inflate_ip(-self.__shrink_width, -self.__shrink_height)

        self._map_size = self._map.calculate_game_plan_size()

        self._movement_speed = 3

        self.last_position = self.get_position()
    
    def _move_left(self) -> None:
        self._wanted_direction = Vector2(0, 0)
        if(self.rect.x > self._map_size[0]):
            self._wanted_direction.x = -1
        
    def _move_right(self) -> None:
        self._wanted_direction = Vector2(0, 0)
        if (self.rect.x < (self._map_size[1] - self.rect.width)):
            self._wanted_direction.x = 1
    
    def _move_up(self) -> None:
        self._wanted_direction = Vector2(0, 0)
        if (self.rect.y > self._map_size[2]):
            self._wanted_direction.y = -1
    
    def _move_down(self) -> None:
        self._wanted_direction = Vector2(0, 0)
        if (self.rect.y < (self._map_size[3] - self.rect.height - 5)):
            self._wanted_direction.y = 1
    
    def _stop_move(self) -> None:
        self._wanted_direction = Vector2(0, 0)

    def _get_direction(self) -> str:
        if self._direction.x > 0:
            return 'right'
        elif self._direction.x < 0:
            return 'left'
        elif self._direction.y > 0:
            return 'front'
        elif self._direction.y < 0:
            return 'back'
        else:
            return 'front'
        
    def handle_animation_state(self) -> None:
        latest_state = self.states[self._current_state.handle_event(self._direction)]
        if not self._current_state == latest_state:
            self._current_state = latest_state
            self._current_frame = 0
        
    def set_dead(self) -> None:
        if not self._current_state == self.states['Dying']:
            self._direction = Vector2(0, 0)
            self._current_frame = 0
            self._current_state = self.states['Dying']
        
    def animate(self, game_display: pygame.display, delta_time):
        self._current_delta_time += delta_time
         
        if self._current_delta_time >= self._animation_speed:
            if self._current_state.get_name() == 'Dying':
                self._current_frame = len(self._all_actions[self._current_state.get_name()]['front']) - 1
            elif self._current_frame < len(self._all_actions[self._current_state.get_name()][self._get_direction()]) - 1:
                self._current_frame += 1
            else:
                self._current_frame = 0
                
            self._current_delta_time = 0
        
        current_image = self._all_actions[self._current_state.get_name()][self._get_direction()][self._current_frame]
        
        game_display.blit(current_image, (self.x, self.y))
            
    def _move_horizontal(self):
        if (self._direction.x == -1 and self.rect.x > self._map_size[0]) or (self._direction.x == 1 and self.rect.x < (self._map_size[1] - self.rect.width)):
            self.x += self._direction.x * self._movement_speed
            self.rect.x = self.x + (self.__shrink_width // 2)
        else:
            self._direction.x = 0
      
    def _horizontal_collisions(self, collidables) -> bool:
        collided = False
        
        for tile in collidables:
            if tile.colliderect(self.rect):
                if self._direction.x < 0:
                    self.rect.left = tile.right
                    self.x = self.rect.x - (self.__shrink_width // 2)
                            
                elif self._direction.x > 0:
                    self.rect.right = tile.left
                    self.x = self.rect.x - (self.__shrink_width // 2)
                    
                collided = True
            
        return collided
    
    def _move_vertical(self):
        if (self._direction.y == -1 and self.rect.y > self._map_size[2]) or (self._direction.y == 1 and self.rect.y < (self._map_size[3] - self.rect.height - 5)):
            self.y += self._direction.y * self._movement_speed
            self.rect.y = self.y + self.__shrink_height / 2
        else:
            self._direction.y = 0
        
    def _vertical_collisions(self, collidables) -> bool:
        collided = False
        
        for tile in collidables:
            if tile.colliderect(self.rect):
                if self._direction.y > 0:
                    self.rect.bottom = tile.top
                    self.y = self.rect.y - self.__shrink_height
                elif self._direction.y < 0:
                    self.rect.top = tile.bottom
                    self.y = self.rect.y - self.__shrink_height
                collided = True
        return collided
    
    def check_position(self, tile) -> bool:
        return (int(tile.rect.x) <= int(self.rect.centerx) < int(tile.rect.x) + int(tile.rect.width)) and \
            (int(tile.rect.y) <= int(self.rect.centery) < int(tile.rect.y) + int(tile.rect.height))
    
    def get_position(self) -> tuple:
        for current_row, row in enumerate(self._map.current_map):
            for current_tile, tile in enumerate(row):
                if self.check_position(tile):
                    return (current_tile, current_row)
        return None
    
    def animate(self, game_display: pygame.display, delta_time):
        self._current_delta_time += delta_time
         
        if self._current_delta_time >= self._animation_speed:
            if self._current_frame < len(self._all_actions[self._current_state.get_name()][self._get_direction()]) - 1:
                self._current_frame += 1
            elif self._current_state.get_name() == 'Dying':
                self._current_frame = len(self._all_actions[self._current_state.get_name()]['front']) - 1
            elif self._current_state.get_name() == 'Idle':
                self._current_frame = len(self._all_actions[self._current_state.get_name()]['front']) - 1
            else:
                self._current_frame = 0
                
            self._current_delta_time = 0

        current_image = self._all_actions[self._current_state.get_name()][self._get_direction()][self._current_frame]
        
        game_display.blit(current_image, (self.x, self.y))

    def _get_accurate_tile_size(self):
        posLeft,posRight,posTop,posBottom = self._map_size
        tile_width = (posRight - posLeft) / self._map._columns
        tile_height = (posBottom - posTop) / self._map._rows

        return tile_width,tile_height,posLeft,posTop,posRight,posBottom
    
    def _get_position_in_grid(self, x,y, tile_width,tile_height,posLeft,posTop):
        r_x = round((x - posLeft) / tile_width)
        r_y = round((y - posTop) / tile_height)
        return r_x,r_y
    
    def _update_is_in_center(self):
        pass
    
    def update(self, game_display: pygame.display, delta_time) -> None:
        position = self.get_position()
        if position is not None:
            x,y = position
            tile_width,tile_height,posLeft,posTop,_,_ = self._get_accurate_tile_size()

            if abs(x*tile_width - (self.x - posLeft)) < 3.1 and abs(y*tile_height - self.y + posTop) < 3.1:
                self._update_is_in_center()
                dirX = x + int(self._wanted_direction.x)
                dirY = y + int(self._wanted_direction.y)
                self.last_position = position
                if (dirX >= 0 and dirX < self._map._columns) and (dirY >= 0 and dirY < self._map._rows):
                    tile = type(self._map.current_map[dirY][dirX])
                else:
                    tile = Wall

                for bomb in self._map.bombs:
                    b_x,b_y = self._get_position_in_grid(bomb.x,bomb.y,tile_width,tile_height,posLeft,posTop)
                    if b_x == dirX and b_y == dirY:
                        tile = Wall
                        break

                if tile == Wall or tile == Crate:
                    self._wanted_direction = Vector2(0, 0)
                self._direction = self._wanted_direction
                self.handle_animation_state()


        self.animate(game_display, delta_time)
        # pygame.draw.rect(game_display, (255, 0, 0), self.rect, 2)