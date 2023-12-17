from pyboy import PyBoy
import os

pyboy = PyBoy('roms/kirby2.gb', rewind = True)
while not pyboy.tick():
    pass
pyboy.stop()
