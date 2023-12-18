from pyboy import PyBoy
from KirbyGame import KirbysDreamland2
import os
import numpy as np
import matplotlib.pyplot as plt

print("Current Working Directory:", os.getcwd())
print("Files in Directory:", os.listdir('.'))

kirby = KirbysDreamland2('roms/kirby2.gb', rewind = True, debug = True, window_type="headless")
frame = 0
window = []
kirby.enable_invincibility()
while not kirby.tick():
    if frame % 10 == 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(kirby)
    frame += 1
    if kirby.paused:
        current_observation = kirby.get_observation()
        plt.imshow(current_observation.screen)
        plt.axis('off')  # To turn off axis labels
        plt.show()
        plt.title('Current window')
        plt.close()
        kirby._unpause()

kirby.stop()