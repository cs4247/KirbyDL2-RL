import datetime
from pathlib import Path
from pyboy.pyboy import *
from gym.wrappers import FrameStack, NormalizeObservation
#from MetricLogger import MetricLogger
from learning_agent import AIKirby
import sys
from CustomGym import KirbyGymEnv

file_like_object = open("save_states/kirby2_2-1.gb.state", "rb")
env = KirbyGymEnv(file_like_object)
env.kirby_game.set_emulation_speed(0)

episodes = 40000
gameDimentions = (144, 160)
frameStack = 4

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
gameName = env.kirby_game.cartridge_title()
save_dir = Path("checkpoints") / gameName / now
save_dir.mkdir(parents=True)
# save_dir_eval = Path("checkpoints") / gameName / (now + "-eval")
# save_dir_boss = Path("checkpoints") / gameName / (now + "-boss")
# checkpoint_dir = Path("checkpoints") / gameName

actions = env.get_actions()
print(f"Actions: {actions}")

player = AIKirby((frameStack,gameDimentions[0],gameDimentions[1]), len(actions), save_dir, now)