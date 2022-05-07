import pygame, random, sys, os
from os import path




WIDTH = 800
HEIGHT = 600
FPS = 30

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255,255,0)


#for score
score_file = path.join(path.dirname(__file__),"highest_score.txt")

def high_score(score,count,total):
    dir = path.dirname(__file__)
    filebuffer = [score,count,float(float(score)/float(count)),float(float(total)/float(count))]
    with open(path.join(dir,"highest_score.txt"),'r') as f:
        try:
            highscore = int(f.read())
        except:
            highscore = 0
    with open(path.join(dir,"highest_score.txt"),'w') as f:
        for value in filebuffer:
            f.write("%s\n" % str(value))
            #f.write("%d\n" for file in filebuffer)





#text for stroop
game_color = [RED,GREEN,BLUE,YELLOW]
game_text = ["RED" , "GREEN" , "BLUE" , "YELLOW" ]
game_key = {"RED":pygame.K_r , "GREEN":pygame.K_g , "BLUE": pygame.K_b , "YELLOW" : pygame.K_y}

#create text label
size= 84


# initialize pygame, this must be called before
# doing anything with pygame
pygame.init()

# create a screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# setup the text
font = pygame.font.Font(None, size)
font_color = random.choice(game_color)
text = font.render(random.choice(game_text), True, font_color)

score=0
score_text = font.render(str(score), True, WHITE)

display = True

keystate = pygame.key.get_pressed()
#for score note ticks
ticks = 0 #  in milli second
time_score = 0;
total =0;
count =1;
total_count=0
# the main loop
while ticks < 60000: # run the program for 60 seconds
     # empty the screen
     screen.fill(BLACK)
     keystate = pygame.key.get_pressed()
     ticks = pygame.time.get_ticks()

     if keystate[pygame.K_r]:
        if font_color == RED:
            score+=1
            total += (pygame.time.get_ticks()-time_score)
        time_score = pygame.time.get_ticks()
        high_score(score,count,total)
        #display = not display
        count+=1
        total_count+=1
        font_color = random.choice(game_color)
        text = font.render(random.choice(game_text), True, font_color)
        score_text = font.render(str(score), True, WHITE)


     if keystate[pygame.K_y]:
        if font_color == YELLOW:
            score+=1
            total += (pygame.time.get_ticks()-time_score)
        time_score = pygame.time.get_ticks()
        high_score(score,count,total)
        #display = not display
        count+=1
        total_count+=1
        font_color = random.choice(game_color)
        text = font.render(random.choice(game_text), True, font_color)
        score_text = font.render(str(score), True, WHITE)


     if keystate[pygame.K_g]:
        if font_color == GREEN:
            score+=1
            total += (pygame.time.get_ticks()-time_score)
        time_score = pygame.time.get_ticks()
        high_score(score,count,total)
        #display = not display
        count+=1
        total_count+=1
        font_color = random.choice(game_color)
        text = font.render(random.choice(game_text), True, font_color)
        score_text = font.render(str(score), True, WHITE)


     if keystate[pygame.K_b]:
        if font_color == BLUE:
            score+=1
            total += (pygame.time.get_ticks()-time_score)
        time_score = pygame.time.get_ticks()
        high_score(score,count,total)
        #display = not display
        count+=1
        total_count+=1
        font_color = random.choice(game_color)
        text = font.render(random.choice(game_text), True, font_color)
        score_text = font.render(str(score), True, WHITE)


     # draw the text to the screen only if display is True
     
     #display = not display
     if display:
         screen.blit(text, (WIDTH/2-80,200))
         screen.blit(score_text , (WIDTH/2,HEIGHT-100))

     # update the actual screen
     pygame.display.flip()
     for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False
                ticks = 1000000

     # wait for half second
     pygame.time.wait(170)
