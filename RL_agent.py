import time
import random
import numpy as np
import json
from Obstacles.crate import Crate
from Obstacles.wall import Wall
from Obstacles.tile import Tile
from Entities.player import Player
from Entities.ai_player import AiPlayer
from bomb import Bomb
from map import Map
import pygame
from States.Entity.idling_state import IdlingState
from Utilities.settings import *


class RL_agent:
    def __init__(self, map: Map, player: AiPlayer) -> None:
        self.player = player
        self.map = map
        self.__old_state = self.__get_state()
        self.discount = 0.6
        self.learning_rate = 0.1
        self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'IDLE', 'PLACE_BOMB']
        self.observations = ["Player","Wall","Crate","Tile","Bomb"]                             #player, wall, crate, tile, bomb
        self.placed_bomb = False
        self.last_decision_time = time.time()
        self.last_action = 'UP'
        self.last_action_with_movement = 'UP'
        self.__died = False
        self.__was_idle = False

    def __init_Q_table(self):
        return {}

    def getFromQT(self, Q_table:dict[list], key:str) -> list:
        if key in Q_table:
            return Q_table[key]
        else:
            return [0 for i in range(len(self.actions))]
        
    def setIntoQT(self, Q_table:dict[list], key:str, data:list):
        if sum(data) == 0:
            return
        Q_table[key] = data
    
    def save_Q_table(self, Q_table, epsilon):
        with open("Q_table.json", "w") as f:
            json.dump({"Q_table":Q_table,"Epsilon":epsilon}, f)

    def load_Q_table(self):
        try:
            with open("Q_table.json", "r") as f:
                data = json.load(f)
                Q_table = data["Q_table"]
                epsilon = data["Epsilon"]
        except FileNotFoundError:
            Q_table = self.__init_Q_table()
            epsilon = 1.0
            self.save_Q_table(Q_table, epsilon)
        return Q_table, epsilon

    def __get_map_with_players(self):
        map_with_players = []
        for line in self.map.current_map:
            tiles = []
            for tile in line:
                tiles.append(type(tile).__name__)
            map_with_players.append(tiles)
        def is_inside_map(x, y):
            return x >= 0 and x < len(self.map.current_map[0]) and y >= 0 and y < len(self.map.current_map)
        for player in self.map.get_players():
            if player != self.player:
                x,y = self.map.calculate_position_on_map(player.x, player.y)
                if is_inside_map(x,y):
                    map_with_players[y][x] = type(player).__name__.replace("AiPlayer","Player")

        for bomb in self.map.bombs:
            x,y = self.map.calculate_position_on_map(bomb.x, bomb.y)
            map_with_players[y][x] = type(bomb).__name__

        return map_with_players
    
    def __get_state(self):
        map_copy = self.__get_map_with_players()        
        tile_width,tile_height,posLeft,posTop,_,_ = self.player._get_accurate_tile_size()
        x,y = self.player._get_position_in_grid(self.player.x,self.player.y,tile_width,tile_height,posLeft,posTop)

        def is_inside_map(x, y):
            return x >= 0 and x < len(map_copy[0]) and y >= 0 and y < len(map_copy)
        on_right = map_copy[y][x + 1] if is_inside_map(x+1,y) else "Wall"
        on_left = map_copy[y][x - 1] if is_inside_map(x-1,y) else "Wall"
        on_top = map_copy[y - 1][x] if is_inside_map(x,y-1) else "Wall"
        on_bottom = map_copy[y + 1][x] if is_inside_map(x,y+1) else "Wall"
        bomb = "NoBomb"
        for map_x in range(x - 2, x + 2):
            if not is_inside_map(map_x,y):
                continue
            if map_copy[y][map_x] == "Bomb":
                bomb = "BombLeft" if map_x < x else "BombRight"
                
        for map_y in range(y - 2, y + 2):
            if not is_inside_map(x,map_y):
                continue
            if map_copy[map_y][x] == "Bomb":
                bomb = "BombTop" if map_y < y else "BombBottom"
        
        for map_x in range(x - 2, x + 2):
            for map_y in range(y - 2, y + 2):
                if not is_inside_map(map_x,map_y):
                    continue
                if map_copy[map_y][map_x] == "Bomb":
                    if map_x < x and map_y < y:
                        bomb = "BombTopLeft"
                    elif map_x < x and map_y > y:
                        bomb = "BombBottomLeft"
                    elif map_x > x and map_y < y:
                        bomb = "BombTopRight"
                    elif map_x > x and map_y > y:
                        bomb = "BombBottomRight"

        if is_inside_map(x,y) and map_copy[y][x] == "Bomb":
            bomb = "InBomb"
        # print(f"{on_left}_{on_right}_{on_top}_{on_bottom}_{bomb}")
        return f"{on_left}_{on_right}_{on_top}_{on_bottom}_{bomb}"

    def __get_info_from_state(self, actions:str):
        left,right,top,bottom,bomb = self.__old_state.split("_")
        directions = {
            'UP' : top,
            'DOWN' : bottom,
            'LEFT' : left,
            'RIGHT' : right
        }
        for action in actions:
            if action in directions.keys():
                return directions[action]
        
    def __is_in_radius(self, type) -> int:
        # b_x = 0
        # b_y = 0
        # player_x, player_y = self.player.last_position
        # map_copy = self.__get_map_with_players()
        # for line in map_copy:
        #     for tile in line:
        #         if tile == "Bomb":
        #             if line.index(tile) == player_x and map_copy.index(line) == player_y:
        #                 b_x = line.index(tile)
        #                 b_y = map_copy.index(line)

        left,right,top,bottom,_ = self.__old_state.split("_")
        sides = [left,right,top,bottom]
        count = 0
        for side in sides:
            if side == type:
                count += 1
        return count



    def __get_reward(self):
        # TODO: 
        # make reward system
        # make bigger radius around player
        # make center of player
        # make standing in bomb debuff
        # make identificator for more players
        # inheritate from ai player
        # rename variables of "old_state" to "performed_state" and "state" to "future_state" or something like that
        # rename variables of "old_action" to "performed_action" and "action" to "future_action" or something like that
        # fix animation of idle

        # +
        # reward for placing bomb next to crate or player ----------------------> 10
        # walking into tile ----------------------------------------------------> 10
        # walking into player --------------------------------------------------> 10
        # walking out of bomb

        # -
        # walking into wall, bomb
        # placing bomb next to wall--------------------------------------------> -10
        # standing in bomb ----------------------------------------------------> -10
        # walking into bomb
        # walking into wall ---------------------------------------------------> -5
        # idling---------------------------------------------------------------> -10
        # dying ---------------------------------------------------------------> -20
        _,_,_,_,bomb = self.__old_state.split("_")
        total_reward = 0    
        if self.__get_info_from_state(['LEFT','RIGHT', 'UP','DOWN']) in ["Tile", "Player"]:
            total_reward += 10
        if self.__get_info_from_state(['LEFT','RIGHT', 'UP','DOWN']) in ["Wall", "Bomb","Crate"]:
            total_reward -= 5
        count_crate = self.__is_in_radius("Crate")
        if self.last_action == 'PLACE_BOMB' and count_crate > 0:
            total_reward += count_crate * 10
        count_player = self.__is_in_radius("Player")
        if self.last_action == 'PLACE_BOMB' and count_player > 0:
            total_reward += count_player * 10
        count_wall = self.__is_in_radius("Wall")
        if self.last_action == 'PLACE_BOMB' and count_wall > 0 and count_crate == 0 and count_player == 0:
            total_reward -= count_wall * 10
        count_bomb = self.__is_in_radius("Bomb")
        if self.last_action == 'IDLE' and count_bomb > 0:
            total_reward -= 10
        if self.last_action == 'IDLE' and bomb == "InBomb":
            total_reward -= 10
        if self.last_action != 'IDLE':
            self.__was_idle = False
        if self.last_action == 'IDLE' and not self.__was_idle:
            self.__was_idle = True
        if self.last_action == 'IDLE' and self.__was_idle:
            total_reward -= 10
        if self.player._current_state == self.player.states['Dying']:
            total_reward -= -20
        
        return total_reward

    def __do_action(self, action):
        if self.placed_bomb:
            self.placed_bomb = False
        match action:
            case 'UP':
                # print("UP")
                self.player._move_up()
            case 'DOWN':
                # print("DOWN")
                self.player._move_down()
            case 'LEFT':
                # print("LEFT")
                self.player._move_left()
            case 'RIGHT':
                # print("RIGHT")
                self.player._move_right()
            case 'PLACE_BOMB':
                # print("PLACE_BOMB")
                self.player.place_bomb()
                self.placed_bomb = True
            case 'IDLE':
                # print("IDLE")
                self.player._stop_move()

    def update(self):
        if self.__died:
            return
        Q_table, epsilon = self.load_Q_table()
        current_time = time.time()
        # for row in self.__get_map_with_players():
        #     out = ""
        #     for tile in row:
        #         out += tile + " "
        #     print(out)  
        # print("---------------------------------------------------------------------------------------")

        state = self.__get_state()
        if current_time - self.last_decision_time >= 0.25:
            random_n = random.random()
            if random_n < epsilon:
                if self.player._can_place_bomb():
                    selected_action = random.choice(self.actions)
                    print("\033[91mRANDOM CHOICE\033[0m")
                else:
                    selected_action = random.choice(self.actions[:-1])
                    print("\033[91mRANDOM CHOICE WITHOUTH BOMB\033[0m")
                

            else:
                if self.player._can_place_bomb():
                    max_action_index = np.argmax(self.getFromQT(Q_table, state))
                    print("\033[94mGREEDY CHOICE\033[0m")
                else:
                    max_action_index = np.argmax(self.getFromQT(Q_table, state)[:-1])
                    print("\033[94mGREEDY CHOICE WITHOUTH BOMB\033[0m")
                value = self.getFromQT(Q_table, state)[max_action_index]
                random_actions = list(filter(lambda index: value - 1 <= self.getFromQT(Q_table, state)[index], range(len(self.getFromQT(Q_table, state)))))
                max_action_index = random.choice(random_actions)
                # print("ACTION INDEX", max_action_index)
                selected_action = self.actions[max_action_index]

            next_state = state
            reward = self.__get_reward()   
            old_value = self.getFromQT(Q_table, self.__old_state)[self.actions.index(self.last_action)]
            next_max = np.max(self.getFromQT(Q_table, next_state))

            new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount * next_max)
            old = self.getFromQT(Q_table, self.__old_state)
            old[self.actions.index(self.last_action)] = new_value
            self.setIntoQT(Q_table, self.__old_state, old)

            print(f"\033[92mCurrent state:{state}, Action: {selected_action}\033[0m")
            # print("Last state:", self.__old_state, "Last action:", self.last_action)
            # print("LAST POSITION", self.player.last_position)
            
            if self.player._current_state == self.player.states['Dying']:
                self.__died = True
                return
            
            self.__do_action(selected_action)
            self.last_decision_time = current_time
            self.last_action = selected_action
            self.__old_state = state

            if epsilon > 0.4:
                epsilon -= epsilon * 0.0001

            self.save_Q_table(Q_table, epsilon)
            print("---------------------------------------------------------------------------------------")
            
