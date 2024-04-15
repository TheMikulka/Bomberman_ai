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
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PLACE_BOMB']
        self.observations = ["Player","Wall","Crate","Tile","Bomb"]                             #player, wall, crate, tile, bomb
        self.placed_bomb = False
        self.last_decision_time = time.time()
        self.last_action = 'UP'
        self.last_action_with_movement = 'UP'
        self.__died = False

    def __init_Q_table(self):
        Q_table = {}
        for bomb in ["BombLeft", "BombRight", "BombTop", "BombBottom", "BombTopLeft", "BombTopRight","BombBottomLeft","BombBottomRight","NoBomb","InBomb"]:
            for left in self.observations:
                for right in self.observations:
                    for top in self.observations:
                        for bottom in self.observations:
                            combination = f"{left}_{right}_{top}_{bottom}_{bomb}"
                            Q_table[combination] = [0 for i in range(len(self.actions))]
        return Q_table
    
    def save_Q_table(self, Q_table, epsilon):
        with open("Q_table.json", "w") as f:
            json.dump({"Q_table":Q_table,"Epsilon":epsilon}, f)

    def load_Q_table(self):
        try:
            with open("Q_table.json", "r") as f:
                data = json.load(f)
                Q_table = data["Q_table"]
                epsilon = data["Epsilon"]
                # print("Q_table loaded successfully")
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
        x, y = self.map.calculate_position_on_map(self.player.x, self.player.y)
        print("X:",x,"Y:",y)
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

        print("STATE:",f"{on_left}_{on_right}_{on_top}_{on_bottom}_{bomb}")
        return f"{on_left}_{on_right}_{on_top}_{on_bottom}_{bomb}"
    
    def __get_informations_from_state(self):
        left,right,top,bottom,bomb = self.__old_state.split("_")
        if self.last_action == 'UP':
            self.last_action_with_movement = 'UP'
            return top, bomb
        if self.last_action == 'DOWN':
            self.last_action_with_movement = 'DOWN'
            return bottom, bomb
        if self.last_action == 'LEFT':
            self.last_action_with_movement = 'LEFT'
            return left, bomb
        if self.last_action == 'RIGHT':
            self.last_action_with_movement = 'RIGHT'
            return right, bomb
        if self.last_action == 'PLACE_BOMB':
            if self.last_action_with_movement == 'UP':
                return top, bomb
            if self.last_action_with_movement == 'DOWN':
                return bottom, bomb
            if self.last_action_with_movement == 'LEFT':
                return left, bomb
            if self.last_action_with_movement == 'RIGHT':
                return right, bomb

    def __get_reward(self):
        state, bomb = self.__get_informations_from_state()
        total_reward = 0
        if bomb != "NoBomb":
            if self.last_action in ['LEFT','RIGHT','DOWN','UP'] and bomb == "InBomb":
                if state == "Tile":
                    print ("BONUS IN BOMB")
                    total_reward += 50
                else:
                    print ("PENALTY IN BOMB")
                    total_reward += -10
            elif self.last_action == 'PLACE_BOMB' and bomb == "InBomb":
                total_reward += -100
            elif self.last_action in ['LEFT','RIGHT','DOWN'] and bomb == "BombUp":
                if state == "Tile":
                    print ("BONUS BOMB UP")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB UP")	
                    total_reward += -10
            elif self.last_action in ['UP','RIGHT','LEFT'] and bomb == "BombDown":
                if state == "Tile":
                    print ("BONUS BOMB DOWN")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB DOWN")
                    total_reward += -10
            elif self.last_action in ['UP','DOWN','LEFT'] and bomb == "BombRight":
                if state == "Tile":
                    print ("BONUS BOMB RIGHT")
                    total_reward += 50
                else:
                    print("PENALTY BOMB RIGHT")
                    total_reward += -10
            elif self.last_action in ['UP','DOWN','RIGHT'] and bomb == "BombLeft":
                if state == "Tile":
                    print ("BONUS BOMB LEFT")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB LEFT")
                    total_reward += -10
            elif self.last_action in ['DOWN','RIGHT'] and bomb == "BombTopLeft":
                if state == "Tile":
                    print ("BONUS BOMB TOP LEFT")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB TOP LEFT")
                    total_reward += -10
            elif self.last_action in ['DOWN','LEFT'] and bomb == "BombTopRight":
                if state == "Tile":
                    print ("BONUS BOMB TOP RIGHT")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB TOP RIGHT")
                    total_reward += -10
            elif self.last_action in ['UP','RIGHT'] and bomb == "BombBottomLeft":
                if state == "Tile":
                    print ("BONUS BOMB BOTTOM LEFT")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB BOTTOM LEFT")
                    total_reward += -10
            elif self.last_action in ['UP','LEFT'] and bomb == "BombBottomRight":
                if state == "Tile":
                    print ("BONUS BOMB BOTTOM RIGHT")
                    total_reward += 50
                else:
                    print ("PENALTY BOMB BOTTOM RIGHT")
                    total_reward += -10
            else:
                print ("PENALTY BOMB")
                total_reward += -100
        if bomb == "InBomb" and self.last_action == 'PLACE_BOMB':
            print("PENALTY PLACE BOMB")
            total_reward += -40

        if self.player._current_state == self.player.states['Dying']:
            print("PENALTY DYING")
            total_reward += -10000
        if state == "Wall" and self.player._current_state == self.player.states['Walking']:
            print("PENALTY WALL")
            total_reward += -10
        if state == "Crate" and self.player._current_state == self.player.states['Walking']:
            print("PENALTY CRATE")
            total_reward += -10
        if state == "Tile" and self.player._current_state == self.player.states['Walking']:
            print("BONUS TILE")
            total_reward += 50
        if state == "Crate" and self.last_action == 'PLACE_BOMB':
            print("BONUS CRATE PLACE BOMB")
            total_reward += 100
        if state == "Wall" and self.last_action == 'PLACE_BOMB':
            print("PENALTY WALL PLACE BOMB")
            total_reward += -10
        if state == "Player" and self.last_action == 'PLACE_BOMB':
            print("BONUS PLAYER PLACE BOMB")
            total_reward += 1000
        if state == "Player" and self.player._current_state == self.player.states['Walking']:
            print("BONUS PLAYER")
            total_reward += 100
        return total_reward

    # def __get_surrounding_objects(self):
    #     # Získání aktuální pozice agenta
    #     current_position = self.player.position
    #     # Seznam objektů v okolí
    #     surrounding_objects = []

    #     # Procházení okolních polí včetně aktuální pozice agenta
    #     for i in range(current_position[0] - 1, current_position[0] + 2):
    #         for j in range(current_position[1] - 1, current_position[1] + 2):
    #             # Zkontrolujeme, zda jsme uvnitř herního pole
    #             if 0 <= i < self.width and 0 <= j < self.height:
    #                 # Přidání objektu na dané pozici do seznamu
    #                 surrounding_objects.append(self.grid[i][j])

    #     return surrounding_objects

    # def __get_reward(self):
    #     state, bomb = self.__get_informations_from_state()
    #     total_reward = 0

    #     # Základní odměny a tresty za různé stavy a akce
    #     if bomb == "InBomb":
    #         if self.last_action == 'PLACE_BOMB':
    #         # Získání informací o okolních polích
    #         surroundings = self.__get_surrounding_objects()
    #         # Kontrola, zda je vedle bomby crate
    #         if "Crate" in surroundings:
    #             total_reward += 100  # Odměna za umístění bomby vedle crate
    #         elif "Wall" in surroundings:
    #             total_reward += -10
    #         elif "Player" in surroundings:
    #             total_reward += 1000
    #     elif bomb.startswith("Bomb"):
    #         if state == "Tile":
    #             total_reward += 50
    #         else:
    #             total_reward += -10
    #     else:
    #         if state == "Player":
    #             total_reward += 100
    #         elif state == "Wall" or state == "Crate":
    #             total_reward += -10

    #     # Specifické odměny a tresty za umístění bomby a interakce s objekty
    #     if self.last_action == 'PLACE_BOMB':
    #         if state == "Crate":
    #             total_reward += 100
    #         elif state == "Wall":
    #             total_reward += -10
    #         elif state == "Player":
    #             total_reward += 1000

    #     # Odměny a tresty za umístění bomby vedle hráče
    #     if bomb == "InBomb" and self.last_action == 'PLACE_BOMB':
    #         total_reward += -40

    #     # Trest za umírání
    #     if self.player._current_state == self.player.states['Dying']:
    #         total_reward += -10000

    #     return total_reward

    def __do_action(self, action):
        if self.placed_bomb:
            self.placed_bomb = False
        match action:
            case 'UP':
                self.player.make_move('up')
            case 'DOWN':
                self.player.make_move('down')
            case 'LEFT':
                self.player.make_move('left')
            case 'RIGHT':
                self.player.make_move('right')
            case 'PLACE_BOMB':
                self.player.make_move('place_bomb')
                self.placed_bomb = True

    def update(self):
        if self.__died:
            return
        Q_table, epsilon = self.load_Q_table()
        current_time = time.time()
        playerx, playery = self.map.calculate_position_on_map(self.player.x, self.player.y)
        # for row in self.__get_map_with_players():
        #     out = ""
        #     for tile in row:
        #         out += tile + " "
        #     print(out)  
        # print("---------------------------------------------------------------------------------------")

        state = self.__get_state()
        estimated_reward = Q_table[state]
        prev_reward = Q_table[self.__old_state]
        if current_time - self.last_decision_time >= 0.25:
            # if np.random.uniform(0,1) < epsilon:
            #     selected_action = np.random.choice(self.actions)
            #     print("RANDOM CHOICE")
            # else:
            #     selected_action = np.argmax(Q_table[state])
            #     print("GREEDY CHOICE")
            random_n = random.random()
            if random_n < epsilon:
                selected_action = random.choice(self.actions[:-1])
                print("RANDOM CHOICE")

            else:
                max_action_index = np.argmax(Q_table[state])
                selected_action = self.actions[max_action_index]
                selected_action = max(self.actions, key=lambda action: Q_table[state][self.actions.index(action)])
                print("GREEDY CHOICE")
            # index_selected_action = self.actions.index(selected_action)
            # Q_table[self.__old_state][self.actions.index(self.last_action)] = prev_reward[self.actions.index(self.last_action)] + self.learning_rate * (get_reward + self.discount * max(estimated_reward) - prev_reward[self.actions.index(self.last_action)])
            next_state = state
            reward = self.__get_reward()   
            old_value = Q_table[self.__old_state][self.actions.index(self.last_action)]
            next_max = np.max(Q_table[next_state])

            new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount * next_max)
            Q_table[self.__old_state][self.actions.index(self.last_action)] = new_value

            print("\nState: [", self.__old_state, state, "]\nSelected action: [", self.last_action, selected_action,  "]\nReward:", reward)
            
            if self.player._current_state == self.player.states['Dying']:
                self.__died = True
                return
            
            self.__do_action(selected_action)
            self.last_decision_time = current_time
            self.last_action = selected_action
            self.__old_state = state

            if epsilon > 0.4:
                epsilon -= epsilon * 0.001

            self.save_Q_table(Q_table, epsilon)
            
