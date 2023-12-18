from pyboy import PyBoy
from dataclasses import dataclass, field
import numpy as np

@dataclass
class GameState:
    lives_left: int = 0
    health: int = 0
    damage_taken: int = 0
    invincible: bool = False
    score: int = 0
    boss_health: int = 0
    boss_active: bool = False
    game_over: bool = False
    star_pieces: int = 0
    level_id: int = 0
    speed: int = 0
    x_loc: int = 0
    screen: np.ndarray = field(default_factory=lambda: np.zeros((144, 160, 3), dtype=np.uint8))


class KirbysDreamland2(PyBoy):
     
    LIVES_LEFT = 0xa084
    HEALTH = 0xa04c
    SCORE_MSB = 0xdedb
    BOSS_HEALTH = 0xdee4
    BOSS_ACTIVE = 0xaa4c
    STAR_PIECE = 0xdee1
    LEVEL_IDENTIFIER = 0xfb84 #Another level id: 0xec00
    WINDOW_X = 0xda01
    KIRBY_X = 0xa009
    
    def __init__(self, gamerom_file, *, bootrom_file=None, disable_renderer=False, sound=False, sound_emulated=False, cgb=None, randomize=False, **kwargs):
        super().__init__(gamerom_file, bootrom_file=bootrom_file, disable_renderer=disable_renderer, sound=sound, sound_emulated=sound_emulated, cgb=cgb, randomize=randomize, **kwargs)

        self.lives_left = 0
        self.health = 0
        self.damage_taken = 0
        self.invincible = False
        self.score = 0
        self.boss_health = 0
        self.boss_active = False
        self.game_over = False
        self.star_pieces = 0
        self.level_id = 0
        self.speed = 0
        self.x_loc = 0
        self._window_x = 0
        self._char_x = 0
        

    def tick(self):
        tick = super().tick()

        #Score calculation
        num_bytes = 3
        self.score = 0
        for i in range(num_bytes):
            v = self.get_memory_value(KirbysDreamland2.SCORE_MSB + i)
            self.score += (int(v/16) * 10 + v%16)*10**(4-2*i)
        self.score *= 10

        #Invincibility
        last_health = self.health
        current_health = self.get_memory_value(KirbysDreamland2.HEALTH)
        if current_health < last_health:
            self.damage_taken += 1
        if self.invincible:
            self.health = 12
            self.set_memory_value(KirbysDreamland2.HEALTH, 12)
        else:
            self.health = current_health
        
        #Lives
        self.lives_left = self.get_memory_value(KirbysDreamland2.LIVES_LEFT)

        #Game over
        self.game_over = self.lives_left == 0 and self.health == 0

        #Star pieces
        self.star_pieces = self.get_memory_value(KirbysDreamland2.STAR_PIECE)

        #Boss
        self.boss_health = self.get_memory_value(KirbysDreamland2.BOSS_HEALTH)
        self.boss_active = self.get_memory_value(KirbysDreamland2.BOSS_ACTIVE) != 0 and self.get_memory_value(KirbysDreamland2.BOSS_ACTIVE) != 1

        #Level identifier. This is one of many addresses related to current level, but this should be good enough to detect room changes.
        last_level_id = self.level_id
        self.level_id = self.get_memory_value(KirbysDreamland2.LEVEL_IDENTIFIER)

        #reset location varaibles if room changes
        if self.level_id != last_level_id:
            self.handle_reset()

        self._handle_position()
        return tick
    
    def _handle_position(self):
        # Get the current X position of the character and window from the game memory
        current_x = self.get_memory_value(KirbysDreamland2.WINDOW_X)
        current_char_x = self.get_memory_value(KirbysDreamland2.KIRBY_X) - 8

        # Calculate the change in X position, accounting for wrap-around
        delta = current_x - self._window_x
        delta_c = current_char_x - self._char_x
        if delta < -128:  # Large negative delta implies wrap-around from low to high
            delta += 256
        elif delta > 128:  # Large positive delta implies wrap-around from high to low
            delta -= 256

        self._window_x = current_x
        self._char_x = current_char_x
        self.x_loc += delta + delta_c
        self.speed = delta + delta_c

    
    def enable_invincibility(self):
        self.invincible = True
        self.health = 12
        self.set_memory_value(KirbysDreamland2.HEALTH, 12)

    def disable_invincibility(self):
        self.invincible = False

    def handle_reset(self):
        self._window_x = self.get_memory_value(KirbysDreamland2.WINDOW_X)
        self._char_x = self.get_memory_value(KirbysDreamland2.KIRBY_X) - 8
        self.x_loc = 0
        self.damage_taken = 0
        self.game_over = False


    def get_observation(self) -> GameState:
        state = GameState()
        state.lives_left = self.lives_left
        state.health = self.health
        state.invincible = self.invincible
        state.score = self.score
        state.boss_health = self.boss_health
        state.boss_active = self.boss_active
        state.game_over = self.game_over
        state.star_pieces = self.star_pieces
        state.level_id = self.level_id
        state.speed = self.speed
        state.x_loc = self.x_loc
        state.screen = self.botsupport_manager().screen().screen_ndarray()
        return state


    def __str__(self):
        return (f"Kirby's Dreamland 2 Game State:\n"
                f"Lives Left: {self.lives_left}\n"
                f"Star pieces: {self.star_pieces}\n"
                f"Health: {self.health}\n"
                f"Damage taken: {self.damage_taken}\n"
                f"Invincible: {self.invincible}\n"
                f"Score: {self.score}\n"
                f"Boss Health: {self.boss_health}\n"
                f"Boss Active: {self.boss_active}\n"
                f"Game Over: {self.game_over}\n"
                f"Level ID: {self.level_id}\n"
                f"Speed: {self.speed}\n"
                f"X Location: {self.x_loc}\n"
                f"Window X: {self._window_x}\n"
                f"Character X: {self._char_x}")