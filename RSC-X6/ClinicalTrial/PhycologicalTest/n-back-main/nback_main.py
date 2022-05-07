"""
pygame-menu
https://github.com/ppizarror/pygame-menu

"""

__all__ = ['main']

import pygame
import pygame_menu
from pygame_menu.examples import create_example_window
import nback

from typing import Tuple, Optional

# Constants and global variables
FPS = 60
WINDOW_SIZE = (1200, 768)
nback.GAME_SELECTED = 0
settings = []                   # creating empty list for the settings, this list will later be used to store the settings to a file
settings_data = ()              # dictionary for storing key pairs of each setting read from the settings file

sound: Optional['pygame_menu.sound.Sound'] = None
surface: Optional['pygame.Surface'] = None
main_menu: Optional['pygame_menu.Menu'] = None


def main_background() -> None:
    """
    Background color of the main menu, on this function user can plot
    images, play sounds, etc.
    """
    surface.fill((40, 40, 40))


# this will restore "factory" settings by copying from one file to another
def restore_factory() -> None:
    with open('factory_settings.txt', 'r') as f1, open('settings.txt', 'w') as f2:
        f2.write(f1.read())                                     # writing the contents of file1 into file2
    f1.close
    f2.close
    main()


# different game modes

def start_single_game() -> None:
    nback.nback("single")

def start_multi_games() -> None:
    nback.nback("multi")

def start_random_multi_games() -> None:
    nback.nback("random_multi")


# --------------------------------------------------------------------------------
#               M A I N
# --------------------------------------------------------------------------------


def main(test: bool = False) -> None:
    """
    Main program.

    :param test: Indicate function is being tested
    """

    # -------------------------------------------------------------------------
    # Globals
    # -------------------------------------------------------------------------
    global main_menu
    global sound
    global surface

    # -------------------------------------------------------------------------
    # Create window
    # -------------------------------------------------------------------------
    surface = create_example_window('N-back Memory Test  -  Made by Thomas Vikström', WINDOW_SIZE)
    clock = pygame.time.Clock()

    # -------------------------------------------------------------------------
    # Set sounds
    # -------------------------------------------------------------------------
    sound = pygame_menu.sound.Sound()

    # Load example sounds
    sound.load_example_sounds()

    # Disable a sound
    sound.set_sound(pygame_menu.sound.SOUND_TYPE_ERROR, None)

    # -------------------------------------------------------------------------
    # Create menus: Settings
    # -------------------------------------------------------------------------
    settings_menu_theme = pygame_menu.themes.THEME_SOLARIZED.copy()
    settings_menu_theme.title_offset = (5, -2)
    settings_menu_theme.widget_alignment = pygame_menu.locals.ALIGN_LEFT
    settings_menu_theme.widget_font = pygame_menu.font.FONT_OPEN_SANS_LIGHT
    settings_menu_theme.widget_font_size = 18

    settings_menu = pygame_menu.Menu(
        height=WINDOW_SIZE[1] * 0.85,
        theme=settings_menu_theme,
        title='Settings',
        width=WINDOW_SIZE[0] * 0.9
    )

    
    # Copying settings into a backup file, just in case the universe crashes :-)
    with open('settings.txt', 'r') as f1, open('settings_backup.txt', 'w') as f2:
        f2.write(f1.read())                                     # writing the contents of file1 into file2
    f1.close
    f2.close

    # Reading settings file for creating and filling in values in the Settings menu
    # as an exercise for the reader, s/he can replace this with Python's CSV-functions :D
    
    f = open('settings.txt', 'r')                               # reading file for menu text inputs
    f.readline()                                                # 1st line has comment, hence reading and then ignoring it

    for line in f:
        fields = line.split(",")                                # splitting line from CSV-file into separate elements  

        id = (fields[0])                                        # the unique ID of each setting
        settings.append(id)                                     # appending to the settings list

        label = (fields[1])                                     # the label text that is shown including a leading space
        settings.append(label)

        max_len = int(fields[2])                                # max length of the input field
        settings.append(max_len)
        
        value = (fields[3][:-1])                                # the value itself, [:-1] is to remove the newline character, this needs to be the last on the line
        settings.append(value)
        settings.append(-1)                                     # to signal we've reached the end of ONE setting

        # Finally creating the menu element based on above parameters
        settings_menu.add.text_input(
            label,
            default = value,
            maxchar = max_len,
            textinput_id = id,
            input_underline='.'
        )

    f.close

    settings_data = settings_menu.get_input_data()              # loading settings from the menu...
    nback.settings_data = settings_data                         # ...and passing them to the "game"


    # -----------------------------------
    # ---------- FUNCTIONS --------------
    # -----------------------------------

    # READING settings from a menu
    def read_setting(id):                                       # reading settings from a menu
        settings_data = settings_menu.get_input_data()          # loading settings into dictionary...
        return settings_data[id]                                # ...and returning the value with id = id

    # SAVING settings to a file
    # as an exercise for the reader, s/he can replace this with Python's CSV-functions :D

    def save_settings() -> None:                                # saving settings to a file

        f = open('settings.txt', 'w')                           # opening file to write

        # header line, for information only, not used for anything
        f.write("-->  Fields are in this order: ID, Label, MaxChar, Value !!! DON'T USE COMMA IN LABELS, DON'T CHANGE THIS FILE unless you know what you are doing !!!\n")

        lines = []                                              # list to construct a line to write
        first_value = True                                      # used later for signalling beginning of line

        for value in settings:
            if first_value == True:                             # if this was the first value on the "line"...
                id = value                                      # ...then store away the value (=name of setting)
            if (value) != -1:                                   # -1 is the last instance
                lines.append(value)                             # appending the values read from the settings file...
                lines.append(',')                               # ...using comma as separator
                first_value = False                             # after this, not first value anymore in this line
            else:
                lines.pop()                                     # removing last comma as we don't want the line to end with a comma
                lines.pop()                                     # also removing the value we read from the settings file...
                lines.append(read_setting(id))                  # ...as it is replaced by the current value read from the menu 
                lines.append('\n')                              # adding new line
                first_value = True                              # end of line

        for line in lines:                                      # writing all lines to the settings file
            f.write(str(line))                                  # converting numbers to strings

        f.close

        # need to pass the changed settings to the "game"
        settings_data = settings_menu.get_input_data()          # loading settings from the menu...
        nback.settings_data = settings_data                     # ...and passing them to the "game"

    # -----------------------------------
    # --------- END FUNCTIONS -----------
    # -----------------------------------


    # Add final divider and buttons
    settings_menu.add.button('_______________________________________________________________________________________', 
                            align=pygame_menu.locals.ALIGN_CENTER)
    settings_menu.add.button('Save settings', save_settings, button_id='Save settings', align=pygame_menu.locals.ALIGN_CENTER)
    settings_menu.add.button('Undo changes', settings_menu.reset_value, align=pygame_menu.locals.ALIGN_CENTER)
    settings_menu.add.button('Restore factory settings', restore_factory, align=pygame_menu.locals.ALIGN_CENTER)
    settings_menu.add.button('Return to main menu', pygame_menu.events.BACK,
                             align=pygame_menu.locals.ALIGN_CENTER)

    
    # -------------------------------------------------------------------------
    # Create menus: Column buttons
    # -------------------------------------------------------------------------
    button_column_menu_theme = pygame_menu.themes.THEME_SOLARIZED.copy()
    button_column_menu_theme.background_color = pygame_menu.BaseImage(
        image_path=pygame_menu.baseimage.IMAGE_EXAMPLE_GRAY_LINES,
        drawing_mode=pygame_menu.baseimage.IMAGE_MODE_REPEAT_XY
    )
    button_column_menu_theme.widget_font_size = 22

    button_column_menu = pygame_menu.Menu(
        columns=2,
        height=WINDOW_SIZE[1] * 0.45,
        rows=3,
        theme=button_column_menu_theme,
        title='Textures+Columns',
        width=WINDOW_SIZE[0] * 0.9
    )
    for i in range(4):
        button_column_menu.add.button(f'Button {i}', pygame_menu.events.BACK)
    button_column_menu.add.button(
        'Return to main menu', pygame_menu.events.BACK,
        background_color=pygame_menu.BaseImage(
            image_path=pygame_menu.baseimage.IMAGE_EXAMPLE_METAL
        )
    ).background_inflate_to_selection_effect()


    # -------------------------------------------------------------------------
    # Create menus: Main menu
    # -------------------------------------------------------------------------
    main_menu_theme = pygame_menu.themes.THEME_SOLARIZED.copy()
    main_menu_theme.title_font = pygame_menu.font.FONT_COMIC_NEUE
    main_menu_theme.widget_font = pygame_menu.font.FONT_COMIC_NEUE
    main_menu_theme.widget_font_size = 30

    main_menu = pygame_menu.Menu(
        height=WINDOW_SIZE[1] * 0.7,
        onclose=pygame_menu.events.EXIT,  # User press ESC button
        theme=main_menu_theme,
        title='Main menu',
        width=WINDOW_SIZE[0] * 0.8
    )

    
    main_menu.add.button('Play single game', start_single_game)
    main_menu.add.button('Play multi games', start_multi_games)
    main_menu.add.button('Play random multi games', start_random_multi_games)
    main_menu.add.button('Settings', settings_menu)
    main_menu.add.button('Quit', pygame_menu.events.EXIT)
    main_menu.add.button('________________________________', )

    # Add a clock
    main_menu.add.clock(clock_format='%d.%m.%Y %H:%M:%S', title_format='Clock: {0}')


    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while True:

        # Tick
        clock.tick(FPS)

        # Paint background
        main_background()

        # Main menu
        main_menu.mainloop(surface, main_background, disable_loop=test, fps_limit=FPS)

        # Flip surface
        pygame.display.flip()

        # At first loop returns
        if test:
            break


if __name__ == '__main__':
    main()