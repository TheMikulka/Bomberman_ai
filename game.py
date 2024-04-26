import pygame, time
from Utilities.settings import *
from States.Game.title_state import TitleState
from States.Game.running_state import RunningState

class Game():
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.FPS = 60
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Bomberman')
        self.fullscreen = False
        self.__monitor_resolution = [pygame.display.Info().current_w, pygame.display.Info().current_h]
        self.__clock = pygame.time.Clock()

        self.font = pygame.font.Font('Assets/Fonts/VCR_OSD_MONO_1.001.ttf', (35 * self.height // self.height))
        
        self.__last_time = time.time()
        
        self.game_states = []
        self.create_states()

    def handle_fullscreen(self):
        self.screen = pygame.display.set_mode(self.__monitor_resolution, pygame.FULLSCREEN)
                            
    def create_states(self) -> None:
        self.game_states.append(TitleState(self))
    
    def set_to_running_state(self) -> None:
        RunningState(self).enter_state()
                
    def run(self) -> None:
        while True:
            current_time = time.time()
            delta_time = current_time - self.__last_time
            self.__last_time = current_time
            
            self.game_states[-1].update(delta_time)
            
            self.__clock.tick(self.FPS)
            pygame.display.flip()