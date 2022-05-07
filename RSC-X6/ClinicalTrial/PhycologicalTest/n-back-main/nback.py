#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#		   			N - B A C K                    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#       Created 2022 from different modules        #
#               by Thomas Vikström                 #
#__________________________________________________#


# Importing modules
import datetime, time
import os
import pygame, random
import math
import csv						# to handle CSV-files
from os.path import exists		# to check if a file exists


log_header = ['Person ID', 
				'Level',
				'Remember both correct/incorrect?',
				'Timestamp shown', 
				'Timestamp clicked',
				'Clicked key',
				'Expected key', 
				'Correctly clicked?',
				'N-back nr', 
				'Shown nr (Nr 0)',
				'Nr 1', 
				'Nr 2',
				'Nr 3',
				'Nr 4']
log_data = []

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialisations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pygame.init()
pygame.font.init()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  C O N S T A N T S  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Keyboard and mouse related
key_press_wait = 0.25									# sleep time before checking if a button was pressed
LEFT_MOUSE = 1
RIGHT_MOUSE = 3
ENTER = 13
ESC = 27
SPACE = 32

# Used for signalling correctly remembered number or not
NONE = "NONE"
NO   = "NO"
YES  = "YES"

game_score = multi_game_score = multi_game_max = 0
breaker = False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#												F U N C T I O N S
# ____________________________________________________________________________________________________________


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reading settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_setting(id):                                       			# reading settings from a menu
	return settings_data[id]                                			# ...and returning the value with id = id


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clears the screen ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def clear_screen():
	screen = pygame.display.set_mode(size, pygame.FULLSCREEN)			# clearing screen
	pygame.display.update()												# update screen
   	

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Shows a random picture ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_picture():

	picture_x = int(read_setting('Pict_x'))								# picture position x
	picture_y = int(read_setting('Pict_y'))								# picture position y
	picture_size_x = int(read_setting('Pict_size_x'))					# picture size x
	picture_size_y = int(read_setting('Pict_size_y'))					# picture size y
	
	random_pic = random.randint(0, len(pictures) - 1)					# randomly selecting a picture
	image = pygame.image.load(picture_folder+'/'+ pictures[random_pic])	# concatenating the folder with pic name

	IMAGE_SIZE = (picture_size_x, picture_size_y)						# setting the size for the picture
	image = pygame.transform.scale(image, IMAGE_SIZE)					# scaling the picture
	IMAGE_POSITION = (picture_x, picture_y)								# placing the picture
	screen.blit(image, IMAGE_POSITION)									# ...and showing it
	pygame.display.update()


# ~~~~~~~~~~~~~~~~~~ Displays a huge white number on black screen ~~~~~~~~~~~~~~~~~~~~
def show_number(n, slide_pause):										# n = number to show, slide_pause = display time
	global screen

	screen = pygame.display.set_mode(size, pygame.FULLSCREEN)			# clearing screen
	pygame.display.update()												# update screen
	time.sleep(slide_pause)												# ZZzzzz
	textsurface = LARGE_FONT.render(str(n[0]), False, (255, 255, 255))	# using a huge font, size defined in the settings file
	text_rect = textsurface.get_rect()
	text_rect.left = xd/2 - text_rect[2]/2
	text_rect.top = yd/2 - text_rect[3]/2
	screen.blit(textsurface, text_rect)

	if use_pictures == 'YES':											# should a picture be shown as well?
		show_picture()
	else:
		pygame.display.update()											# if not, only update screen
	



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Displays a header ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_text(text):													# text = text to show
	screen = pygame.display.set_mode(size, pygame.FULLSCREEN)			# clearing screen
	textsurface = MEDIUM_FONT.render(text, False, (20, 200, 20))		# using a medium font, size defined in the settings file
	text_rect = textsurface.get_rect()
	text_rect.left = xd/2 - text_rect[2]/2	
	text_rect.top = yd/15
	screen.blit(textsurface, text_rect)
	pygame.display.update()												# update screen


# ~~~~~~~~~~~~~~~~~~~~ Halts the program until a key or mouse button is pressed ~~~~~~~~~~~~~~~~~~~~
def wait_for_key_press(seconds):
	while 1:
		time.sleep(seconds)
		ev = pygame.event.get()
		if ev != [] and (ev[0].type == pygame.MOUSEBUTTONDOWN 
				or ev[0].type == pygame.KEYDOWN):								# see e.g. https://riptutorial.com/pygame/example/18046/event-loop
			break


# ~~~~~~~~ Checks if a mouse button, SPACE, or ENTER is pressed, does not halt the program ~~~~~~~~~
# ~~~~~~~~ --> returns the button or key pressed
def button():
	ev = pygame.event.get()
	for event in ev:
		if event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == LEFT_MOUSE or event.button == RIGHT_MOUSE:		# LEFT or RIGHT mouse button
				return int(event.button)
		elif event.type == pygame.KEYDOWN:
			key = event.key
			if key == pygame.K_SPACE or key == pygame.K_RETURN or key == pygame.K_ESCAPE :		# SPACE/ENTER/ESC
				return int(key)


# ~~~~~~~~~~~~~~~~~~~~~ Returns the current time, shorter to use ... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def now():
	return str(datetime.datetime.now())											# ...than this 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Playing a sound sound ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sound():
	pygame.mixer.Sound.play(start_sound)
	pygame.mixer.music.stop()


# 													E N D    F U N C T I O N S					
# _______________________________________________________________________________________________________________________________



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 						T H E   G A M E                                							
# _______________________________________________________________________________________________________________________________

def nback(mode):															# mode is single, multi, multi random ...
	global level, screen, xd, yd, size										# level=1..4, xd & yd=resolution, size=(xd,yd)
	global settings_data													# settings dictionary
	global breaker															# used to quit games
	global MEDIUM_FONT, LARGE_FONT											# font sizes
	global start_sound														# storing the sound
	global use_pictures														# using pictures or not
	global pictures															# list for storing picture file names
	global picture_folder													# picture folder, what else :D


	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 										S U B   F U N C T I O N S
	# __________________________________________________________________________________________________


	# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reading settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# def read_setting(id):                                       			# reading settings from a menu
	# 	return settings_data[id]                                			# ...and returning the value with id = id


	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Playing a level ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	def play_level(level):
		pygame.display.set_caption('N-back game, level: '+ str(level))		# showing caption and level
		clear_screen()

		# breathing
		
		# show_text('Level ' + str(level) + ' next: Click when ready')		# removed upon request
		# wait_for_key_press(key_press_wait)

		show_text('Deep breathing next: Click when ready')
		wait_for_key_press(key_press_wait)
		show_text('Deep breathing, eyes closed')
		time.sleep(breathing_time)											# ZZZzzzzz

		# ready to start...
		sound()																# ding...dong...
		if level == 0:
			show_text('Level 0, target nr: '+str(_0_back_target)+'. Click when ready')	# showing level chosen + target nr
		else:
			show_text('Level '+str(level)+' will start : Click when ready')	# showing level chosen
		wait_for_key_press(key_press_wait)

		# ...almost there, initializing...
		slides = int(read_setting(str(level) + '_slides'))					# reading setting for nr of slides (= numbers) to show
		n = [0, 0, 0, 0, 0]													# number queue
		game_score = 0


		# ...and finally starting the main loop
		for i in range(0, slides):											# the main loop 
			n[4] = n[3]														# moving numbers 1 step back, can be rewritten with...
			n[3] = n[2]														# ...another for-loop, but this is easier to understand
			n[2] = n[1]
			n[1] = n[0]
			dice = random.randrange(0, 100 // prev_nr_probability + 1)		# % chance to see "previous" n-level number...
			if dice == 0:													# 
				n[0] = n[level]												# ... i.e. same as n-back (1...4) 
			else:
				n[0] = random.randrange(0, 10)								# random number 0-9
			

			# level 0 needs special treatment
			if level == 0:
    				
				dice = random.randrange(0, 100 // prev_nr_probability + 1)	# % chance to see the target number...
				if dice == 0:												# if we "got lucky"...
					n[0] = _0_back_target									# the target number is what to show...
					expected = _0_back_target								# ...and thus the expected nr...
					expected_key = "Left/Space"								# ...one of these is what to click
				else:
					n[0] = random.randrange(0, 10)							# random number 0-9
					if n[0] != _0_back_target:
						expected = -1
						expected_key = "Right/Enter"						# ...else one of these	
					else:
						expected = _0_back_target							# if we happened to get the target number by chance
						expected_key = "Left/Space"							# ...one of these is what to click...

			# levels 1-4
			else:
				expected = n[level]											# expected number for levels 1-4
				if expected == n[0]:										# if shown number = expected...
					expected_key = "Left/Space"								# ...one of these is what to click...
				else:
					expected_key = "Right/Enter"							# ...else one of these				
			
			# print(n)
			# print(expected)												# debug info

			time_stamp_shown = now()										# storing away current time
			show_number(n, pause_between)									# showing the "magic" number
			time.sleep(slide_pause)											# how long to show each slide

			btn = button()													# looking if a button/key was pressed
			if btn == ESC:													# if ESC pressed, we want to ASAP end the misery...
				breaker = True												# ...dirty? way to jump out of several loops 
				break
			else:
				breaker = False

			correct = NONE
			time_stamp_clicked = ''
			clicked_key = ''

			if i > (level - 1):												# no need to check for the first numbers shown (as per level), also would mess up scoring
				if btn == LEFT_MOUSE or btn == SPACE:						# checking if left mouse or <SPACE> was pressed
					time_stamp_clicked = now()
					clicked_key = "Left/Space"

					if n[0] == expected:									# did the user remember correctly?
						correct = YES
						game_score += 1										# increase game score
					else:
						correct = NO

				elif remember_both == 'YES':								# here the user should remember both if shown nur was same or not same
					if btn == RIGHT_MOUSE or btn == ENTER:					# was right mouse or <ENTER> pressed
						time_stamp_clicked = now()
						clicked_key = "Right/Enter"

						if n[0] != expected:								# did the user remember correctly?
							correct = YES
							game_score +=1									# increase game score
						else:
							correct = NO

					elif n[0] == expected:									# if the user didn't click, but should have...
						game_score -= 0										# ...decreasing score
					elif n[0] != expected and expected != -1:				# if the user didn't click, and shouldn't have clicked...
						game_score +=1										# ...increasing score

				elif remember_both == 'NO':
					if n[0] != expected:									# if the user didn't click, and shouldn't have clicked...
						game_score +=1										# ...increasing score


			# WRITING TO CSV_FILE
			# creates the row to write to the CSV-file
			log_data = [person, level, remember_both, time_stamp_shown, time_stamp_clicked, clicked_key, expected_key, correct, expected]
			for x in range(0,len(n)):
				log_data.append(n[x])										# appending the number queue
			
			if not exists(f_name):											# writing header only if file does NOT exist...
				with open(f_name, 'w', encoding='UTF8', newline='') as f:	# creates a new file
					writer = csv.writer(f)
					writer.writerow(log_header)								# writes HEADER
					writer.writerow(log_data)								# writes the data to the csv file

			else:															# ...as we don't want a header between every data row
				with open(f_name, 'a', encoding='UTF8', newline='') as f:	# existing file, opens for appending to it
					writer = csv.writer(f)
					writer.writerow(log_data)								# writes the data to the csv file

		max_score = slides - level											# maximum score is nr of slides - played level

		return [game_score, max_score, breaker]								# ---> ending the function by returning a list (breaker used...
																			# ... to signal forward that the user pressed ESC)

	# 											E N D   S U B   F U N C T I O N S					
	# ______________________________________________________________________________________________________________________



	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#												S T A R T   O F   n b a c k
	# ______________________________________________________________________________________________________________________


	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reading screen related settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	xd = int(read_setting('ResX'))											# what is the X-resolution?
	yd = int(read_setting('ResY'))											# what is the Y-resolution?
	size = (xd, yd)															# creating a tuple of them both

	medium_font_size = int(read_setting('Header_font_size'))				# header font size
	MEDIUM_FONT = pygame.font.SysFont('Comic Sans MS', medium_font_size)

	large_font_size = int(read_setting('Number_font_size'))					# number font size
	LARGE_FONT = pygame.font.SysFont('Comic Sans MS', large_font_size)




	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reading game related settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	person = read_setting('Person')											# person ID, used e.g. in the log file name
	level = int(read_setting('Level'))										# level
	remember_both = read_setting('Remember_both').upper()					# expecting YES or NO
	breathing_time = int(read_setting('Breath_time'))						# breathing time in seconds
	slide_pause = float(read_setting('Slide_pause'))						# number display time in seconds
	pause_between = float(read_setting('Pause_between'))					# pause between slides in seconds
	pause_games = float(read_setting('Pause_games'))						# pause between games in seconds
	multi_games = read_setting('Multi_games')								# multi games, levels
	random_multi = read_setting('Random_games_levels')						# random multi games, levels
	random_multi_N = int(read_setting('Random_games_N'))					# how many random games to play
	_0_back_target = int(read_setting('0_back_target'))						# for 0-back, which is the target number (PS variables cannot start with a number, hence _ in the beginning)
	show_scores = read_setting('Show_scores').upper()						# expecting YES or NO

	prev_nr_probability = int(read_setting('prev_nr_probabil'))				# probability to see previous n-level number
	if prev_nr_probability == 0:
		prev_nr_probability = 1												# used to avoid division by zero 

	f_name = person + read_setting('Log_file')								# log file to write to, person ID added before

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sound related settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	sound_file = read_setting('Start_sound')
	start_sound = pygame.mixer.Sound(sound_file)

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Picture related settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	use_pictures = read_setting('Use_pictures')								# want to show pictures or not? (YES/NO)
	if use_pictures == 'YES':
		picture_folder = read_setting('Picture_folder')						# from which folder (only pictures allowed, other files will cause very interesting "side effects")
		pictures = os.listdir(picture_folder)								# fetching ALL files from the folder into the list


	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Choosing single/multi/random multi game ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if mode == 'single':													# single game
		game_score = play_level(level)										# play only the chosen level

		# scoring
		percent = game_score[0] / game_score[1] * 100						# score in %
		if show_scores == 'YES':
			show_text(f"Score: {game_score[0]}/{game_score[1]} ({percent:.0f} %). Click when ready.")
			wait_for_key_press(key_press_wait)

	elif mode == 'multi':													# multi games
		games = multi_games.split("/")                                		# splitting field into separate elements
		multi_game_score = 0
		multi_game_max = 0

		for game in games:													# play every game found in the list
			game_score = play_level(int(game))								# playing the game

			# calculating scores
			multi_game_score += game_score[0]								# accumulating achieved score from one game
			multi_game_max += game_score[1]									# accumulating max score from one game

			clear_screen()
			if game_score[2] == True:										# ESCaping out of the game
				break
			time.sleep(pause_games)											# pausing between games

		# scoring
		percent = multi_game_score / multi_game_max * 100					# score in %
		if show_scores == 'YES':
			show_text(f"Score: {multi_game_score}/{multi_game_max} ({percent:.0f} %). Click when ready.")
			wait_for_key_press(key_press_wait)

	elif mode == 'random_multi':											# random multi games
		game_levels = random_multi.split("-")								# splitting field into a list
		lvl_start = int(game_levels[0])										# playing from this level...
		lvl_end = int(game_levels[1])										# ...until this

		multi_game_score = 0
		multi_game_max = 0

		for game in range(random_multi_N):									# play N amount of games
			random_game = random.randrange(lvl_start, lvl_end + 1)			# randomly selecting what game to play
			game_score = play_level(int(random_game))						# playing the random game

			# calculating scores
			multi_game_score += game_score[0]								# accumulating achieved score from one game
			multi_game_max += game_score[1]									# accumulating max score from one game

			clear_screen()
			if game_score[2] == True:										# ESCaping out of the game
				break
			time.sleep(pause_games)											# pausing between games

		# scoring
		percent = multi_game_score / multi_game_max * 100					# score in %
		if show_scores == 'YES':
			show_text(f"Score: {multi_game_score}/{multi_game_max} ({percent:.0f} %). Click when ready.")
			wait_for_key_press(key_press_wait)


# ____________________________________________  E N D   O F   P R O G R A M  _______________________________________________