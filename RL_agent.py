import time
import random
import numpy as np
import math
import json
from Obstacles.crate import Crate
from Obstacles.wall import Wall
from Obstacles.tile import Tile
from Entities.player import Player
from Entities.ai_player import AiPlayer
from bomb import Bomb
from map import Map
from Q_table import Q_table
import pygame
from States.Entity.idling_state import IdlingState
from Utilities.settings import *



class RL_agent(AiPlayer):
    def __init__(self, coords: tuple, entity_name: str, n_frames: tuple, s_width: int, s_height: int, scale, map: Map, game_display: pygame.display, identifier: int, table: Q_table) -> None:
        super().__init__(coords, entity_name, n_frames, s_width, s_height, scale, map, game_display, identifier)
        self.performed_state = self.__get_state()
        self.discount = 0.6
        self.learning_rate = 0.1
        self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'IDLE', 'PLACE_BOMB']
        self.observations = ["Player","Wall","Crate","Tile","Bomb"]                             #player, wall, crate, tile, bomb
        self.last_decision_time = time.time()
        self.performed_action = None
        self.__died = False
        self.__waiting_for_died_calculation = False
        self.table = table
        self.__last_choice_method = ""

    def __get_map_with_players(self):
        map_with_players = []
        for line in self._map.current_map:
            tiles = []
            for tile in line:
                tiles.append(type(tile).__name__)
            map_with_players.append(tiles)
        def is_inside_map(x, y):
            return x >= 0 and x < len(self._map.current_map[0]) and y >= 0 and y < len(self._map.current_map)
        for player in self._map.get_players():
            if player != self and player._current_state != player.states['Dying']:
                x,y = self._map.calculate_position_on_map(player.x, player.y)
                if is_inside_map(x,y):
                    map_with_players[y][x] = type(player).__name__.replace("AiPlayer","Player").replace("RL_agent","Player")

        for bomb in self._map.bombs:
            x,y = self._map.calculate_position_on_map(bomb.x, bomb.y)
            map_with_players[y][x] = type(bomb).__name__

        return map_with_players
    
    def _calculate_nearest_player(self):
        players = self._map.get_players()
        smallest_distance = 1000000000000000000000
        closest_player = None
        for player in players:
            if player != self and player._current_state != player.states['Dying']:
                distance = math.dist([self.x,self.y],[player.x,player.y])
                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_player = player
        return closest_player
    
    def __determine_direction_nearest_player(self):
        nearest_player = self._calculate_nearest_player()
        # if nearest_player is not None:
        #     print(f"{self._identifier}:{nearest_player._identifier}")
        if nearest_player is not None:
            distance_x = abs(self.x - nearest_player.x)
            distance_y = abs(self.y - nearest_player.y)
            if distance_x > distance_y:
                return 'EnemyOnLeft' if self.x > nearest_player.x else 'EnemyOnRight'
            else:
                return 'EnemyOnTop' if self.y > nearest_player.y else 'EnemyOnBottom'
        return 'NoEnemy'
    
    def __get_state(self):
        map_copy = self.__get_map_with_players()        
        tile_width,tile_height,posLeft,posTop,_,_ = self._get_accurate_tile_size()
        x,y = self._get_position_in_grid(self.x,self.y,tile_width,tile_height,posLeft,posTop)

        def is_inside_map(x, y):
            return x >= 0 and x < len(map_copy[0]) and y >= 0 and y < len(map_copy)
        on_right = map_copy[y][x + 1] if is_inside_map(x+1,y) else "Wall"
        on_left = map_copy[y][x - 1] if is_inside_map(x-1,y) else "Wall"
        on_top = map_copy[y - 1][x] if is_inside_map(x,y-1) else "Wall"
        on_bottom = map_copy[y + 1][x] if is_inside_map(x,y+1) else "Wall"

        on_right_top = map_copy[y - 1][x + 1] if is_inside_map(x+1,y-1) else "Wall"
        on_right_bottom = map_copy[y + 1][x + 1] if is_inside_map(x+1,y+1) else "Wall"
        on_left_top = map_copy[y - 1][x - 1] if is_inside_map(x-1,y-1) else "Wall"
        on_left_bottom = map_copy[y + 1][x - 1] if is_inside_map(x-1,y+1) else "Wall"

        on_right_right = map_copy[y][x + 2] if is_inside_map(x+2,y) else "Wall"
        on_left_left = map_copy[y][x - 2] if is_inside_map(x-2,y) else "Wall"
        on_top_top = map_copy[y - 2][x] if is_inside_map(x,y-2) else "Wall"
        on_bottom_bottom = map_copy[y + 2][x] if is_inside_map(x,y+2) else "Wall"

        on_center = map_copy[y][x] 

        nearest_player = self.__determine_direction_nearest_player()
        
        return f"{on_top_top}_{on_left_top}_{on_top}_{on_right_top}_{on_left_left}_{on_left}_{on_center}_{on_right}_{on_right_right}_{on_left_bottom}_{on_bottom}_{on_right_bottom}_{on_bottom_bottom}_{nearest_player}"
        
    def __is_in_radius(self, type) -> int:
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        sides = [left,right,top,bottom]
        count = 0
        for side in sides:
            if side == type:
                count += 1
        return count

    def __is_in_far_radius(self, type) -> int:
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        sides = [left_left, right_right, top_top, bottom_bottom]
        count = 0
        for side in sides:
            if side == type:
                count += 1
        return count
    
    def __get_direction(self):
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        match self.performed_action:
            case 'UP':
                return [center,top,top_top]
            case 'DOWN':
                return [center,bottom,bottom_bottom]
            case 'LEFT':
                return [center,left,left_left]
            case 'RIGHT':
                return [center,right,right_right]
    
    def __get_center(self):
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        return center

    def __get_corners(self):
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        return [left_top,left_bottom,right_top,right_bottom]
    
    def __get_direction_corners(self):
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        match self.performed_action:
            case 'UP':
                return [left_top, right_top]
            case 'DOWN':
                return [left_bottom, right_bottom]    
            case 'LEFT':
                return [left_top, left_bottom]
            case 'RIGHT':
                return [right_top, right_bottom]
    
    def _where_is_neares_player(self):
        top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom,bottom_bottom,nearest_player = self.performed_state.split("_")
        match nearest_player:
            case 'EnemyOnLeft':
                return "LEFT"
            case 'EnemyOnRight':
                return "RIGHT"
            case 'EnemyOnTop':
                return "UP"
            case 'EnemyOnBottom':
                return "DOWN"
            case 'NoEnemy':
                return "NoEnemy"



    def __print_state(self,state):
        #top_top, left_top, top, right_top, left_left, left, center, right, right_right, left_bottom, bottom, right_bottom, bottom_bottom
        #     [0]
        #   [1,2,3]
        # [4,5,6,7,8]
        #   [9,10,11]
        #     [12]
        
        states = state.split("_")
        out = [[] for _ in range(5)]
        for index,s in enumerate(states):
            if index == 0:
                out[index].append(s[0])
            if index in [1,2,3]:
                out[1].append(s[0])
            if index in [4,5,6,7,8]:
                out[2].append(s[0])
            if index in [9,10,11]:
                out[3].append(s[0])
            if index == 12:
                out[4].append(s[0])
                
        print(f"    {".".join(out[0])}\n  {".".join(out[1])}\n{".".join(out[2])}\t {states[13]}\n  {".".join(out[3])}\n    {".".join(out[4])}")
        


    def __get_reward(self):
        # TODO: 

        # make reward system
        # make standing in bomb debuff ------------✔✔✔✔✔
        # make bigger radius around player--------✔✔✔✔✔
        # make center of player-------------------✔✔✔✔✔
        # make identificator for more players-----✔✔✔✔✔
        # inheritate from ai player---------------✔✔✔✔✔
        # rename variables of "old_state" to "performed_state" and "state" to "future_state" or something like that ---------✔✔✔✔✔
        # rename variables of "old_action" to "performed_action" and "action" to "future_action" or something like that -----✔✔✔✔✔
        # fix animation of idle

        # make method for get direction of player when moving --------✔✔✔✔✔
        # if staying in bomb and can move just one tile - --------------------->-10
        # if staying in bomb and can move more than one tile + ----------------> 10
        # if see bomb and walk into it - -------------------------------------->-15

        # +
        # reward for placing bomb next to crate or player ----------------------> 10
        # walking into tile ----------------------------------------------------> 10
        # walking into player --------------------------------------------------> 10
        # walking out of bomb --------------------------------------------------> 10
        # walking into player

        # -
        # walking into wall or crate ------------------------------------------> -5
        # walking into bomb ---------------------------------------------------> -15
        # placing bomb next to wall--------------------------------------------> -10
        # standing in bomb ----------------------------------------------------> -10
        # idling---------------------------------------------------------------> -10
        # dying ---------------------------------------------------------------> -20

        total_reward = []    
        #---------------------------------WALKING---------------------------------
        if self.performed_action in ['LEFT','RIGHT','UP','DOWN']:
            center,close,far = self.__get_direction()
            if close in ["Tile", "Player"]:
                total_reward.append(("walk to tile/player",10))

            if close in ["Wall", "Crate"]:
                total_reward.append(("walk to wall/crate", -5))

            if far in ["Crate", "Player"] and close in ["Tile"] and self.__is_in_radius("Bomb") == 0 and center!="Bomb":
                total_reward.append(("walk into far crate/player",5))

            if center == "Bomb" and close == "Tile":
                if far == "Tile":
                    total_reward.append(("walk out of bomb by tile",10))
                elif "Tile" in self.__get_direction_corners():
                    total_reward.append(("walk out of bomb by corner",10))
                else:
                    total_reward.append(("walk out of bomb, but blocked",-10))
            elif center == "Bomb" and close in ["Wall", "Crate"]:
                total_reward.append(("walk in to the wall/crate when in bomb",-10))

            if center == "Tile" and (close == "Bomb" or far == "Bomb"):
                total_reward.append(("walk into explosion",-25))
            if "Bomb" in self.__get_direction_corners():
                total_reward.append(("walk from corner bomb to explosion",-20))
            if self.__is_in_radius("Bomb") != 0 and close == "Tile":
                total_reward.append(("walk away from bomb", 10))
            
            #---------------------------------WALKING_TO_ENEMY---------------------------------

            if self.performed_action == self._where_is_neares_player() and "Bomb" not in self.__get_direction_corners() and self.__is_in_radius("Bomb") == 0 and center != "Bomb" and close == "Tile":
                total_reward.append(("Walking to enemy",10))
                   
        #---------------------------------PLACING BOMB---------------------------------
        count_crate = self.__is_in_radius("Crate")
        if self.performed_action == 'PLACE_BOMB' and count_crate > 0:
            total_reward.append(("Reward for placing bomb next to crate",count_crate * 10))
        if self.performed_action != 'PLACE_BOMB' and count_crate > 0 and self.__get_center() != "Bomb" and self.__is_in_radius("Bomb") == 0 and self._can_place_bomb() and self.performed_action != self._where_is_neares_player() :
            total_reward.append(("Reward for not placing bomb next to crate",count_crate * -15))

        count_player = self.__is_in_radius("Player")
        if self.performed_action == 'PLACE_BOMB' and count_player > 0:
            total_reward.append(("Reward for placing bomb next to player",count_player * 10))

        count_wall = self.__is_in_radius("Wall")
        if self.performed_action == 'PLACE_BOMB' and count_wall > 0 and count_crate == 0 and count_player == 0:
            total_reward.append(("Reward for placing bomb next to wall",count_wall * -10))

        #---------------------------------IDLE---------------------------------
        count_bomb = self.__is_in_radius("Bomb")
        count_far_bomb = self.__is_in_far_radius("Bomb")
        center_idle = self.__get_center()
        if self.performed_action == 'IDLE' and count_bomb > 0:
            total_reward.append(("Standing next to bomb",-10))
        elif self.performed_action == 'IDLE' and "Bomb" in self.__get_corners():
            total_reward.append(("standing in corners",10))
        elif self.performed_action == 'IDLE' and center_idle == "Bomb":
            total_reward.append(("Standing in bomb",-10))
        elif self.performed_action == 'IDLE' and count_far_bomb > 0:
            total_reward.append(("Standing in far bomb",10))
        elif self.performed_action == 'IDLE' and self._can_place_bomb():
            total_reward.append(("Idling",-5))

        #---------------------------------DYING---------------------------------
        # if self._current_state == self.states['Dying']:
        #     total_reward.append(("Dying",-20))

        print(f"\033[1;35mReward: {total_reward}\033[0m")
        return sum(map(lambda x: x[1], total_reward))

    def __do_action(self, action):
        print(f"\033[1;35mDO ACTION: {action}\033[0m")
        match action:
            case 'UP':
                self._move_up()
            case 'DOWN':
                self._move_down()
            case 'LEFT':
                self._move_left()
            case 'RIGHT':
                self._move_right()
            case 'PLACE_BOMB':
                self._stop_move()
                self.place_bomb()
            case 'IDLE':
                self._stop_move()

    def _update_is_in_center(self):
        self._stop_move()
        self.calculate_next_action()

    
    def update(self, delta_time) -> None:
        if self._current_state == self.states['Dying']:
            self.calculate_next_action()
        return super().update(delta_time)
    
    def calculate_next_action(self):
        if self.__died and not self.__waiting_for_died_calculation:
            return
        current_time = time.time()

        future_state = self.__get_state()
        choosed_method = ""
        if current_time - self.last_decision_time >= 0.1:
            random_n = random.random()
            if random_n < self.table.epsilon:
                # if self.table.epsilon > 0.4:
                if self._can_place_bomb():
                    selected_action = random.choice(self.actions)
                    choosed_method = "\033[91mRANDOM CHOICE\033[0m"
                else:
                    act = self.actions[:-1]
                    selected_action = random.choice(act)
                    choosed_method = "\033[91mRANDOM CHOICE WITHOUTH BOMB\033[0m"
                # else:
                #     selected_action = random.choice(self.actions[:-1])
                #     print("\033[91mRANDOM CHOICE UNDER 0.4 epsilon\033[0m")
            else:
                f_state = []
                if self._can_place_bomb():
                    f_state = self.table.get(future_state)
                    max_action_index = np.argmax(f_state)
                    choosed_method = "\033[94mGREEDY CHOICE\033[0m"
                else:
                    f_state = self.table.get(future_state)[:-1]
                    max_action_index = np.argmax(f_state)
                    choosed_method = "\033[94mGREEDY CHOICE WITHOUTH BOMB\033[0m"
                value = self.table.get(future_state)[max_action_index]
                random_actions = list(filter(lambda index: value - 3 <=f_state[index], range(len(f_state))))
                max_action_index = random.choice(random_actions)
                selected_action = self.actions[max_action_index]

            next_max = np.max(self.table.get(future_state))
            print(self.__last_choice_method)
            # print(choosed_method)

            if self.performed_action is not None:
                reward = self.__get_reward()   
                old_value = self.table.get(self.performed_state)[self.actions.index(self.performed_action)]

                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount * next_max)
                old = self.table.get(self.performed_state)
                old[self.actions.index(self.performed_action)] = new_value
                self.table.set(self.performed_state, old)
                print(f"\033[1;35mFINAL Reward: {reward}\033[0m")
                if self.__waiting_for_died_calculation:
                    self.__waiting_for_died_calculation = False

            print(f"\033[1m\033[94mPLAYER: {self._identifier}\033[0m")
            print(f"Current state: {self.performed_state}")
            print(f"Q table value: {",".join(list(map(lambda x: f"{self.actions[x[0]][0]}: {round(x[1],3)}", enumerate(self.table.get(self.performed_state)))))}")
            self.__print_state(self.performed_state)
            print(f"\033[92mAction: {self.performed_action}\033[0m")
            
            # print(f"\033[1m\033[94mPLAYER: {self._identifier}\033[0m")
            # print(f"Current state: {future_state}")
            # self.__print_state(future_state)
            # print(f"\033[92mAction: {selected_action}\033[0m")


            if self._current_state == self.states['Dying'] and not self.__died:
                self.__died = True
                self.__waiting_for_died_calculation = True
            if self.__died and not self.__waiting_for_died_calculation:
                print(f"\033[1;31mPLAYER: {self._identifier} DIED\033[0m")
                return
            
            self.__do_action(selected_action)
            self.last_decision_time = current_time
            self.performed_action = selected_action
            self.performed_state = future_state
            self.__last_choice_method = choosed_method

            if self.table.epsilon > 0.4:
                self.table.epsilon -= self.table.epsilon * 0.0001

            
            print("---------------------------------------------------------------------------------------")
            
