from pyboy import PyBoy
from KirbyGame import KirbysDreamland2
import os
import numpy as np
import matplotlib.pyplot as plt

print("Current Working Directory:", os.getcwd())
print("Files in Directory:", os.listdir('.'))

kirby = KirbysDreamland2('roms/kirby2.gb', rewind = True, debug = True)
frame = 0
window = []
while not kirby.tick():
    if frame % 10 == 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(kirby)
    frame += 1
    if kirby.paused:
        plt.imshow(kirby.botsupport_manager().screen().screen_ndarray())
        plt.axis('off')  # To turn off axis labels
        plt.show()
        plt.title('Current window')
        kirby._unpause()

kirby.stop()