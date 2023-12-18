import gym
from gym import spaces
import numpy as np
from pyboy import WindowEvent
from KirbyGame import KirbysDreamland2, GameState

class KirbyGymEnv(gym.Env):

    #Image parameters
    HEIGHT = 144
    WIDTH = 160
    CHANNELS = 3
    
    def __init__(self, game_state_file, boss_fight = False):
        super(KirbyGymEnv, self).__init__()
        self.action_space = spaces.Discrete(len(self.get_actions())) # Replace N with the number of actions
        # Using the screen as observation
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(KirbyGymEnv.HEIGHT,KirbyGymEnv.WIDTH, KirbyGymEnv.CHANNELS), 
                                            dtype=np.uint8)
        
        self.kirby_game: KirbysDreamland2 = KirbysDreamland2('roms/kirby2.gb', window_type="headless")
        self.game_state = game_state_file
        self.kirby_game.load_state(game_state_file)
        self.last_state = self.kirby_game.get_observation()
        self.boss_fight = boss_fight

        #From PyBoy/openai_gym
        self._buttons = [
            WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.PRESS_BUTTON_START
        ]
        self._button_is_pressed = {button: False for button in self._buttons}

        self._buttons_release = [
            WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_START
        ]
        self._release_button = {button: r_button for button, r_button in zip(self._buttons, self._buttons_release)}

    
    #The actions are every button press, and every combination of two button presses at once. This is the same action space as PyBoy-RL
    def get_actions(self):
        action_tuples = [
            (WindowEvent.PRESS_BUTTON_A,),
            (WindowEvent.PRESS_BUTTON_B,),
            (WindowEvent.PRESS_ARROW_UP,),
            (WindowEvent.PRESS_ARROW_DOWN,),
            (WindowEvent.PRESS_ARROW_LEFT,),
            (WindowEvent.PRESS_ARROW_RIGHT,),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_DOWN),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_RIGHT),
            (WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_UP),
            (WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_DOWN),
            (WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_LEFT),
            (WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_RIGHT),
            (WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN),
            (WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_LEFT),
            (WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_RIGHT),
            (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT),
            (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_RIGHT),
            (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT)
        ]
        return action_tuples

    def step(self, actions):
        info = {}
        pyboy_done = False

        #Handle button presses in the same way as PyBoy-RL/CustomPyBoyGym
        if actions[0] == WindowEvent.PASS:
            pyboy_done = self.kirby_game.tick()
        else:
            # release buttons if not pressed now but were pressed in the past
            for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
                if pressedFromBefore not in self.get_actions():
                    release = self._release_button[pressedFromBefore]
                    self.kirby_game.send_input(release)
                    self._button_is_pressed[release] = False

            # press buttons we want to press
            for buttonToPress in self.get_actions():
                self.kirby_game.send_input(buttonToPress)
                self._button_is_pressed[buttonToPress] = True # update status of the button

        current_state = self.kirby_game.get_observation()
        reward = self._calculate_reward(current_state, self.game_state)
        game_done = current_state.game_over
        if self.boss_fight:
            game_done = not current_state.boss_active
        else:
            game_done = current_state.boss_active or current_state.level_id == 20 #20 is the minigame after every level
        done = pyboy_done or game_done
        
        observation = current_state.screen

        self.last_state = current_state
        return observation, reward, done, info

    def reset(self):
        self.kirby_game.load_state(self.game_state)
        self.last_state = self.kirby_game.get_observation()
        observation = self.last_state.screen
        return observation

    def render(self, mode='human'):
        print(self.kirby_game)

    def _calculate_reward(self, current_state: GameState, last_state: GameState):
        if not self.boss_fight:

            #lost a life
            if current_state.health == 0 and last_state.health != 0:
                return -1000
            
            #Health rewards
            if current_state.health < last_state.health:
                return -100
            elif current_state.health > last_state.health: #Healed
                return 100
            
            #Gained starpiece, star pieces actually decrease if you get enough to increase your lives left
            if current_state.star_pieces > last_state.star_pieces or current_state.lives_left > last_state.lives_left:
                return 1000

            #Completed level
            if current_state.health > 0 and current_state.level_id == 20 or current_state.boss_active:
                return 1000
            elif current_state.game_over:# died
                return -1000
            
            #Movement rewards
            if current_state.speed >= 3: #Moving right quickly
                return 5
            elif current_state.x_loc > last_state.x_loc: #Moved right
                return 2
            
            if current_state.speed <= -3: #Moving left quickly
                return -5
            elif current_state.x_loc < last_state.x_loc: #Moving left
                return -2
            
            #If no other conditions are met, just decrease the reward to implement a timer-like behavior
            return -1
        else:#This is a boss fight

            #defeated boss
            if current_state.boss_health == 0 and last_state.boss_health != 0:
                return 10000
            #died
            elif current_state.game_over:
                return -10000
            
            #Damage boss
            if current_state.boss_health < last_state.boss_health:
                return 100
            #Got hit
            if current_state.health < last_state.health:
                return -100
            
            return -5 #Decrease reward as time goes on to incentivise quicker episodes

    def close(self):
        pass
