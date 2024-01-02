import pygame
import time

class RingPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.music_file_1 = "./sound/final.mp3"
        self.music_file_2 = "./sound/next-tier.mp3"

    def play_next(self):
        pygame.mixer.music.load(self.music_file_1)
        pygame.mixer.music.play()

    def play_final(self):
        pygame.mixer.music.load(self.music_file_2)
        pygame.mixer.music.play()

    def stop_music(self):
        pygame.mixer.music.stop()
