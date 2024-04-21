import pygame
from Utilities.settings import *
from .game_state import GameState
from .pause_state import PauseState
from Entities.player import Player
from Entities.ai_player import AiPlayer
from Q_table import Q_table
from map import Map
from RL_agent import RL_agent

class RunningState(GameState):
    def __init__(self, game) -> None:
        GameState.__init__(self, game)
        self.initialize()
        
    
    def initialize(self) -> None:
        self.map = Map(self._game.width, self._game.height)
        self.table = Q_table("Q_table.json", 6)
        # self.player1 = Player(self.map.set_starting_postion(0, 0), 'player_1', PLAYER_1_CONTROLS, (8, 7, 4), 32, 32, 2.2, self.map, self._game.screen)
        self.ai = RL_agent(self.map.set_starting_postion(0, 0), 'player_1', (8, 7, 4), 32, 32, 2.2, self.map, self._game.screen, 1,self.table)
        self.ai2 = RL_agent(self.map.set_starting_postion(0, 24), 'player_4', (8, 7, 4), 32, 32, 2.2, self.map, self._game.screen, 2, self.table)
        self.ai3 = RL_agent(self.map.set_starting_postion(12, 0), 'player_2', (8, 7, 4), 32, 32, 2.2, self.map, self._game.screen, 3, self.table)
        self.ai4 = RL_agent(self.map.set_starting_postion(12, 24), 'player_3', (8, 7, 4), 32, 32, 2.2, self.map, self._game.screen, 4, self.table)


        
    def handle_events(self) -> None:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        new_state = PauseState(self._game)
                        new_state.enter_state()
                    if event.key == pygame.K_f:
                        self._game.fullscreen = not self._game.fullscreen
                        if self._game.fullscreen:
                            self.handle_fullscreen()
        
    def update(self, delta_time) -> None:
        self.map.render_map(self._game.screen, delta_time)
        # self._game.draw_text('PLAYER 1', (100 * self._game.width // self._game.width), (20 * self._game.height // self._game.height))
        
        self.handle_events()
        
        pressed_keys = pygame.key.get_pressed()
        count_of_alive = 0
        for player in self.map.get_players():
            if player._current_state != player.states['Dying']:
                count_of_alive += 1

        if count_of_alive <= 1:
            all_players_dead = True
            for player in self.map.get_players():
                if player._current_state != player.states['Dying']:
                    continue
                if player._current_frame != len(player._all_actions[player._current_state.get_name()]['front']) - 1:
                    all_players_dead = False

            if all_players_dead:
                self.table.save()
                self.initialize()
                return
                
        for player in self.map.get_players():
            if isinstance(player, Player):
                player.update(pressed_keys, delta_time)
            else:
                player.update(delta_time)
        
        