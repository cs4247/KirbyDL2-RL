import datetime
from pathlib import Path
from pyboy.pyboy import *
from gym.wrappers import FrameStack, NormalizeObservation
from learning_agent import AIKirby
import sys
from CustomGym import KirbyGymEnv
from metrics import MetricLogger
import time

file_like_object = "save_states/kirby2_2-1.gb.state"
env = KirbyGymEnv(file_like_object)
env.kirby_game.set_emulation_speed(0)

episodes = 40000
gameDimentions = (144, 160)
frameStack = 4

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
gameName = env.kirby_game.cartridge_title()
save_dir = Path("checkpoints") / gameName / now
save_dir.mkdir(parents=True)

actions = env.get_actions()
print(f"Actions: {actions}")

player = AIKirby((4,144,160), len(actions), save_dir, now)
logger = MetricLogger(save_dir)
player.saveHyperParameters()
player.net.train()

for episode in range(episodes):
    print(f"Episode {episode} of {episodes}")
    observation = env.reset()
    start = time.time()
    while True:
        actionId = player.act(observation)
        action = actions[actionId]
        next_observation, reward, done, info = env.step(action)
        
        player.cache(observation, next_observation, actionId, reward, done)
        q, loss = player.learn()
        logger.log_step(reward, loss, q, player.scheduler.get_last_lr())
        observation = next_observation
        if done or time.time() - start > 500:
            break

    logger.log_episode()
    logger.record(episode=episode, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step, maxLength=env.kirby_game.x_loc)
env.close()