import pygame, time
from Utilities.settings import *
from States.Game.title_state import TitleState

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
        
    # def __handle_windowed(self, width: int, height: int) -> None:
    #     self.__screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    #     self.__map.resize_map(width, height)

    def handle_fullscreen(self):
        self.screen = pygame.display.set_mode(self.__monitor_resolution, pygame.FULLSCREEN)
                            
    def draw_text(self, text: str, x: int, y: int) -> None:
        text_surface = self.font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, (text_rect.x, text_rect.y))
        
    def create_states(self) -> None:
        self.game_states.append(TitleState(self))
                
    def run(self) -> None:
        while True:
            current_time = time.time()
            delta_time = current_time - self.__last_time
            self.__last_time = current_time
            
            self.game_states[-1].update(delta_time)
            
            self.__clock.tick(self.FPS)
            pygame.display.flip()