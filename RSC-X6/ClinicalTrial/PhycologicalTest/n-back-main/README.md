# INTRODUCTION to this n-back test platform/game
This test platform/game was developed **solely** for the purpose of assessing working memory capacity and to see if/how increased cognitive workload can be seen in EEG-data. So, don't expect a fancy game with all bells and whistles, but hey - it is working more often than not!
## 1 What is a n-back task?
The n-back task is a continuous performance task that is commonly used as an assessment in psychology and cognitive neuroscience to measure a part of working memory and working memory capacity. The n-back was introduced by Wayne Kirchner in 1958. See https://en.wikipedia.org/wiki/N-back for more information.
## 2 Installation
Prerequisites:
- Python version 3.10.2 however, as nothing really fancy is being used, somewhat older and newer versions should work.

Installation:
- Install pygame and pygame-menu by executing these from a command prompt:
  - pip install pygame
  - pip install pygame-menu
- Copy the files in this repository to a folder of your choice.
- Start the software by running nback_main.py with your favourite Python-IDE or from the command prompt. 
## 3 Instructions for use
The objective is to test and hopefully also improve your working memory capacity over time.

#### Worth to know before testing:
There are 5 levels to choose between, what level/levels you want to test is set in the `Settings` menu.
- Level 0: you are supposed to click `Space` or `Left mouse button` when you see the target number. The target number is set in the settings menu, default is 4.
- Level 1 - 4: You are supposed to click `Space` or `Left mouse button` when the previous n number is repeated. 
  - E.g., if level is 1 and you have seen these numbers: 2, 8, 1, 1 - then you should have reacted once you saw the second 1.
  - If level is 2 and you have seen these numbers: 3, 1, 0, 9, 0, 9 - then you should have reacted when you saw the second 0 and second 9.
#### Testing your memory, quick and easy
Why not start with the easiest i.e., `Play single game`. Unless you have messed with the default settings, you will need to react when seeing number 4.
When the test has ended, you will see your scores. If you get tired during the test, you can abort with `Esc`.
#### Testing your memory, slow and hard
Use `Play multi games` or `Play random multi games`. These work similarly as the single game, but instead of only one game, you will play n amount of games according to the settings. 
Scores will only be shown after the final game. Tired? --> click `Esc`.
See the `Settings` section for more information.
#### Feeling confident? Challenge yourself!
If you want to challenge yourself, then you can adjust the settings so that you also need to react when the number shown is **not** same as the target number (level 0) or previous n number (level 1 - 5). The keys you use for this are `Enter` or `Right mouse button`.
#### Want to see your performance over time?
The program spits out a log file in CSV-format, check this to calculate scores yourself and see your performance over time.

## 4 Settings

There are quite a few settings, and even more can be added in an easy and clever(?) way. Obviously **you** are expected to code the behaviour of the new setting.
Most of the settings are self-explanatory, but here they are anyhow explained:
- **Test person ID**  : unique ID for the test person, can be a name, initials, or whatever
- **Remember both if number was shown/not shown?**  : if YES, then you are expected to react both if the number shown was same (`Space/Left mouse`) or not same (`Enter/Right mouse`) as previous n. Only YES or NO should be stored here (no error handling implemented).
- **Single game: level (0-4)** : Which level to play for the single game?
- **Multi game: levels (separate with /)** : When playing multi games, the levels will be in this order. Guess what, separate the levels with /. Using something else will make you extremely unhappy!
- **Random multi games:   How many to play?** : When playing random multi games, how many would you like to play?...
- **Levels?** : ...and which are the levels? Use e.g. 0-4, 1-2, or whatever. Again, no error control.
- **0-back target number**: When playing level 0, which is the target number the user should react upon?
- **Breathing time (seconds)**: Waiting time before the game starts, this is useful when simultaneously recording EEG-data to see if this relaxing period is visible or not in the data.
- **Showing each slide this many seconds**: A slide = a single number that will be shown. Change this to make the test harder or even more harder. 
- **Pause between each slide (seconds)**: = Pause between each slide (seconds)
- **Pause between each game (seconds)**: Useful in multi game sessions to allow for more relaxation.
- **Nr of slides 0-back -> 4-back**: How many slides (= numbers) should be shown for each level?
- **Probability-% to see previous n-level number**: What is the probability that you will see the previous shown number? E.g., in level 1, and if the previous number shown was 5, probability is set to 25 %, then next number to be shown will be 5 with 25 % probability.
- **Show scores after each session?**: YES if you want to see the score, anything else here means NO.
- **Show pictures from picture folder?**: Do you want to see a smallish picture somewhere on the screen at the same time as a number is shown? Perhaps a picture of a spider or a snake? If you love these animals the game might be easier, otherwise perhaps not... YES or NO.
- **Picture folder (only pictures are allowed in it)**: If previous setting = YES, then this folder will be used to find pictures. Best is to have pictures of same aspect ratio, otherwise they will be stretched. Only pictures are allowed in this folder, having other types of files there will make you very unhappy! Easiest is to create a subfolder in the folder where this program is saved, name it `Pictures`, and then just write `Pictures` in this setting.
- **Picture position - X + Y**: Where on the screen do you want this picture...
- **Picture size - X + Y**: ...and how large should it be?
- **Screen resolution - X + Y**: Depends on your screen, try e.g. with 1200 x 768.
- **Font size - Header**: This is the size of the text shown before a level starts.
- **Font size - Number**: Size of the magic numbers to be shown
- **Starting game sound (file name)**: This is the sound that is played after the breathing period, use a valid file name, otherwise...
- **Log file (Person ID will be added in front):**: Why not leave this as it is? In that case, if your name is e.g. `Amnesia` (or whatever is stored in the setting `Test person ID`, then the resulting log file name will be `Amnesia_log.csv`.

### 4.1 Change the order of the existing settings? Change the text or default value of the settings? More settings needed?
This is only for the bravehearted! Take a backup of `settings.txt` before, as it is this file you will mess around with!
- If you only want to change the order of the settings in the settings menu, use Notepad or whatever text editor and move the settings around. DO NOT move or remove the first line!
- Changing the text of a setting (= what is visible in the settings menu) you do by changing the **second** "column". **Everything** between the first and second comma (,) you can change to whatever text you like, so you can e.g. use spaces creatively as I have done. DO NOT use comma (,) in the text though, this would lead to too many commas in the line and something will for sure break.
- The default value of each setting is last on each line, change this if you want. Remember, **you** are the error handling, not the program!
- Next to last (= third "column") is the maximum amount of characters that can be written from inside the settings menu. If you e.g. need to have a veeery long path to a file or folder, you can increase it here. This setting has no other implications.
- If you can code in Python, you can create and use your own settings. Just add a new line at any point in the file (not as very first line though), create an ID of your choice, e.g. `My_fancy_new_setting` and utilize this new setting in the code. This is the way I myself started from 3 settings and ended up with 30 so far, clever huh?
  - In the code you can search for e.g. `person = read_setting('Person')` to see how it works. The function `read_setting` expects a setting to be read, what you provide as input is the **ID** of your setting, thus in this silly example you would use `my_fancy_new_setting = read_setting('My_fancy_new_setting')`, and then you'd "just" need to add code somehow using this new setting.

### 4.2 Saving/undoing setting changes? Restoring to factory settings?
Scroll to the bottom in the settings menu, and you'll find these more or less self explanatory options:
- Save settings: Use this to save any changes you have done, otherwise the changes will be lost when you return to the main menu or close the program.
- Undo changes: Undo any changes you have done, e.g. if your cat jumped on your keyboard (has happened to me), then this would be the easiest way.
- Restore factory settings: Restores the settings to the ones in the file `factory_settings.txt`.
By the way, every time you save the settings, the old ones are copied to the file `OLD_settings.txt`. If you (or your cat) have really messed around, you can copy this file, or the content of it, to `settings.txt`.

## 5 Improvement recommendations for you to consider doing yourself!
- Error control! 
  - E.g., messing around manually with the settings file - unless you know what you are doing - will cause havoc!
  - Also using invalid log filenames, letters instead of numbers (why would you do that?), or not following the instructions might cause interesting behaviour...
- Supporting letters instead of/in addition to numbers in the game itself
- Now the user needs to change the display resolution completely manually by giving x and y resolution. Change this so the user can select from a list of resolutions supported by the user's hardware.
- Anything and everything else!
  - I learned 80 % of my current Python skills while writing this software, this might be visible in the code, so you are welcome to improve it! 
## Credits
- sound from https://soundbible.com/1598-Electronic-Chime.html
- pygame https://www.pygame.org/news
- pygame-menu https://pygame-menu.readthedocs.io/en/4.2.5/index.html
