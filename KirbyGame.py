from pyboy import PyBoy
import numpy as np

class KirbysDreamland(PyBoy):

    #Memory Adressess
    SCORE_MSB = 0xD06F
    HEALTH = 0xD086
    LIVES_LEFT = 0xD089
    BOSS_HEALTH = 0xD093
    SCREEN_X_POS = 0xD053
    KIRBY_X_POS = 0xD05C
    KIRBY_Y_POS = 0xD05D
    GAME_STATE = 0xD02C

    def __init__(self, gamerom_file, *, bootrom_file=None, disable_renderer=False, sound=False, sound_emulated=False, cgb=None, randomize=False, **kwargs):
        super().__init__(gamerom_file, bootrom_file=bootrom_file, disable_renderer=disable_renderer, sound=sound, sound_emulated=sound_emulated, cgb=cgb, randomize=randomize, **kwargs)

        #assert self.cartridge_title() == "KIRBY DREAM LA"

        self.shape = (20, 16)
        self.score = 0
        self.health = 0
        self.lives_left = 0
        self._game_over = False
        self.fitness = 0
        self.screen_x_position = 0
        self.kirby_x_position = 0
        self.kirby_y_position = 0
        self.boss_health = 0
        self.boss_active = False
        self.invincible = False

    def tick(self):
        tick = super().tick()

        #Parameters from PyBoy-RL KirbyAI Settings 
        self.boss_health = self.get_memory_value(KirbysDreamland.BOSS_HEALTH)
        self.boss_active = self.boss_health > 0
        self.screen_x_position = self.get_memory_value(KirbysDreamland.SCREEN_X_POS)
        self.kirby_x_position = self.get_memory_value(KirbysDreamland.KIRBY_X_POS)
        self.kirby_y_position = self.get_memory_value(KirbysDreamland.KIRBY_Y_POS)
        
        #Parameters from PyBoy Gamewrapper
        self.score = 0
        score_digits = 5
        for n in range(score_digits):
            self.score += self.get_memory_value(KirbysDreamland.SCORE_MSB + n) * 10**(score_digits - n)

        previous_health = self.lives_left
        self.lives_left = self.get_memory_value(KirbysDreamland.LIVES_LEFT) - 1
        if self.lives_left == 0:
                if previous_health > 0 and self.health == 0:
                    self._game_over = True
        
        if self.invincible and self.get_memory_value(KirbysDreamland.HEALTH) < 6:
             self.set_memory_value(KirbysDreamland.HEALTH, self.health)
        else:
            self.health = self.get_memory_value(KirbysDreamland.HEALTH)

        self.fitness = self.score * self.health * self.lives_left
        return tick
    
    def enable_invincibility(self):
         self.invincible = True
         self.health = 6
         self.set_memory_value(KirbysDreamland.HEALTH, self.health)

    def disable_invincibility(self):
         self.invincible = False

    def __str__(self):
        return (
            f"Shape: {self.shape}\n"
            f"Score: {self.score}\n"
            f"Health: {self.health}\n"
            f"Lives Left: {self.lives_left}\n"
            f"Game Over: {self._game_over}\n"
            f"Fitness: {self.fitness}\n"
            f"Screen X Position: {self.screen_x_position}\n"
            f"Kirby X Position: {self.kirby_x_position}\n"
            f"Kirby Y Position: {self.kirby_y_position}\n"
            f"Boss Active: {self.boss_active}\n"
            f"Boss Health: {self.boss_health}\n"
            f"Invincible: {self.invincible}"
        )

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
        self.invincible = False
        self.score = 0
        self.boss_health = 0
        self.boss_active = False
        self._game_over = False
        self.hit_top = False
        self.star_pieces = 0
        self.level_id = 0
        self.speed = 0
        self.x_loc = 0
        self._window_x = 0
        self._char_x = 0
        self.ciel_collision = False

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
        if self.invincible:
            self.health = 12
            self.set_memory_value(KirbysDreamland2.HEALTH, 12)
        else:
            self.health = self.get_memory_value(KirbysDreamland2.HEALTH)
        
        #Lives
        self.lives_left = self.get_memory_value(KirbysDreamland2.LIVES_LEFT)

        #Game over
        self._game_over = self.lives_left == 0 and self.health == 0

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
            self._window_x = self.get_memory_value(KirbysDreamland2.WINDOW_X)
            self._char_x = self.get_memory_value(KirbysDreamland2.KIRBY_X) - 8

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

    #The screen window as a 3d array, a recreation of what it would have done if this was implemented in pyboy's GameWrapper class
    def game_area(self):
        xx = 0
        yy = 0
        width = 18
        height = 18
        scanline_parameters = self.botsupport_manager().screen().tilemap_position_list()
        tilemap_background = self.botsupport_manager().tilemap_background()

        game_area_tiles = np.ndarray(shape=(height, width), dtype=np.uint32)
        for y in range(height):
            SCX = scanline_parameters[(yy+y) * 8][0] // 8
            SCY = scanline_parameters[(yy+y) * 8][1] // 8
            for x in range(width):
                _x = (xx+x+SCX) % 32
                _y = (yy+y+SCY) % 32
                game_area_tiles[y, x] = tilemap_background.tile_identifier(_x, _y)

        return game_area_tiles



    def __str__(self):
        return (f"Kirby's Dreamland 2 Game State:\n"
                f"Lives Left: {self.lives_left}\n"
                f"Star pieces: {self.star_pieces}\n"
                f"Health: {self.health}\n"
                f"Invincible: {self.invincible}\n"
                f"Score: {self.score}\n"
                f"Boss Health: {self.boss_health}\n"
                f"Boss Active: {self.boss_active}\n"
                f"Game Over: {self._game_over}\n"
                f"Hit Top: {self.hit_top}\n"
                f"Level ID: {self.level_id}\n"
                f"Speed: {self.speed}\n"
                f"X Location: {self.x_loc}\n"
                f"Window X: {self._window_x}\n"
                f"Character X: {self._char_x}")