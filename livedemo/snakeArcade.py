import pygame
import pygame_menu
import matplotlib
import sys
import random
import time
from pygame.locals import *
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import threading
import time
import math

from tkinter import Tk, Label
from PIL import Image, ImageTk
from datetime import datetime

#font_path = "Downloads/NotoColorEmoji-Regular.ttf"  # Replace with the path to your font
# Initialize pygame
pygame.init()
# Set up display

n, m = 4, 4
width, height = 640, 480
width, height = 2000, 2000
width, height = 8*80, 8*80; n, m = 4, 4
cell_size = width/n
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption('Snakes - Logic is Everywhere')

# Colors
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

# Clock
clock = pygame.time.Clock()
fps = 10

# Snake and Food
snake = [[1, 1]]
direction = 'None'
food = (random.randrange(0, n) ,
        random.randrange(0, m) )
steps = []
stepstotal = 0
start_time = 0
end_time = 0
times = []
check = False

# Font
font = pygame.font.SysFont(None, 35)
stats_file = 'game_stats.json'

def load_stats():
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return []

def save_stats(stats):
    with open(stats_file, 'w') as f:
        json.dump(stats, f)

def add_stats():
    stats = load_stats()
    stats.append({'n': n, 'm': m, 'l': len(snake), 's': stepstotal, 'ss': steps, 't': sum(times), 'tt': times})
    save_stats(stats)

def calculate_percentage(v, x):
    v_sorted = sorted(v)
    position = 0
    l=0
    for value in v_sorted:
        if math.isnan(value):
            continue
        if x >= value:
            position += 1
        l +=1
    if l>0:
        return (position / len(v)) * 100
    else:
        return 0

def visualize_stats( save = False):
    global check
    stats = load_stats()

    # Filter stats for matching n and m
    tmax = sum(times) +1
    matching_stats = [stat for stat in stats if stat['n'] == n and stat['m'] == m]
    hss = np.array([stat['ss'] for stat in matching_stats])
    hl  = np.array([stat['l'] for stat in matching_stats])
    htt = np.array([stat['tt'] for stat in matching_stats])
    ht  = np.array([stat['t'] if stat['l']==n*m else tmax for stat in matching_stats])
    hs  = np.array([stat['s'] if stat['l']==n*m else stepstotal+1 for stat in matching_stats])
    hl  = np.array([stat['l'] for stat in matching_stats])
    

    hss = np.where(hss == 0.0, np.nan, hss)
    htt = np.where(htt == 0.0, np.nan, htt)
    ss = np.where(steps == 0.0, np.nan, steps)
    tt = np.where(times == 0.0, np.nan, times)


    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)

    def calculate_percentiles(data, percentiles):
        data_no_zeros = np.where(data == 0, np.nan, data)
        return [np.nanpercentile(data_no_zeros, p, axis=0) for p in percentiles]
    
    if matching_stats:
        percentiles = [10, 25, 50, 75, 90]
        p10s, p25s, p50s, p75s, p90s = calculate_percentiles(hss, percentiles)
        p10t, p25t, p50t, p75t, p90t = calculate_percentiles(htt, percentiles)

        ps = calculate_percentage(hl,len(snake))
        if len(snake) < n*m:
            plt.suptitle(f'Snake Statistik für Länge {len(snake)} (Top {int(100-ps-0.49)}%)', fontsize=24)
        else:
            ps = calculate_percentage(hs,sum(times))
            pt = calculate_percentage(ht,sum(times))
            plt.suptitle(f'Snake Statistik für Zeit {int(sum(times))}s (Top {int(100-ps-0.49)}%, kürzeste {int(ps+0.49)}%, schnellste {int(pt+0.49)}%)', fontsize=24)

        # Plot ss
        plt.fill_between(range(len(p10s)), p10s, p90s, color='blue', alpha=0.1, label='10%-90% Perzentile')
        plt.fill_between(range(len(p25s)), p25s, p75s, color='blue', alpha=0.3, label='25%-75% Perzentile')
        plt.plot(p50s, color='blue', linestyle='--', label='Anzahl Schritte im Mittel')
        plt.plot(ss, color='black', label='Aktuelles Spiel')
        plt.title('Anzahl Schritte pro Iteration')
        plt.grid()
        plt.legend()

        # Plot t
        plt.subplot(2, 2, 2)
        plt.title('Akkumulierte Zeit pro Iteration')
        plt.fill_between(range(len(p10t)), p10t, p90t, color='green', alpha=0.1, label='10%-90% Perzentile')
        plt.fill_between(range(len(p25t)), p25t, p75t, color='green', alpha=0.3, label='25%-75% Perzentile')
        plt.plot(p50t, color='green', linestyle='--', label='akkumulierte Zeit im Mittel')
        plt.plot(tt, color='black', label='Aktuelles Spiel')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig("stats.png")
    if save:
        add_stats()
    plt.close()
    check = True



def load_svg(size):    
    # Convert SVG to PNG and load it
    cairosvg.svg2png(url="/home/elisa/Dokumente/shared/snake/output/eyes.svg", write_to="temp_eyes.png", output_width=size, output_height=size)
    cairosvg.svg2png(url="/home/elisa/Dokumente/shared/snake/output/apple.svg", write_to="temp_apple.png", output_width=size, output_height=size)

def draw_svg(typ, position):
    x, y = position
    image = pygame.image.load("temp_"+typ+".png")
    screen.blit(image, (x, y))

# Game over function
def game_over():
    end_time = time.time()
    duration = int(end_time - start_time)
    visualize_stats(True)
    # screen.fill(black)
    # if len(snake) ==n*m:
    #     game_over_text = font.render(f'Won! L: {len(snake)} S: {(stepstotal)} T: {duration}s', True, green)
    # else:
    #     game_over_text = font.render(f'Game Over! L: {len(snake)} S: {(stepstotal)} T: {duration}s', True, red)
    # screen.blit(game_over_text, (width // 6, height // 2))
    # pygame.display.flip()
    # waiting_for_input = True
    # while waiting_for_input:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
    #         elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.KEYDOWN:
    #             waiting_for_input = False
    #             break
    #     time.sleep(0.1)
    start_menu()

def display_plot():
    root = Tk()
    root.title("Statistics Viewer")
    
    img = Image.open("stats.png")
    photo = ImageTk.PhotoImage(img)
    label = Label(root, image=photo)
    label.image = photo  # Keep a reference!
    label.pack()
    
    def update_image():
        global check
        while True:
            if check:
                time.sleep(0.1)  # Refresh every second
                img = Image.open("stats.png")
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo)
                label.image = photo
                check = False
            time.sleep(0.5)  # Refresh every second
            #print(len(snake))
    
    threading.Thread(target=update_image, daemon=True).start()
    
    root.mainloop()

def save_screenshot(screen):
    path = "/home/elisa/Dokumente/shared/snake/output/"
    if not os.path.exists(path+"img"):
        os.makedirs(path+"img")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"{path}img/screenshot_{current_time}.png"
    pygame.image.save(screen, screenshot_filename)
    print(f"Screenshot saved as {screenshot_filename}")

# Draw grid background
def draw_grid():
    screen.fill((255, 255, 255))
    for x in range(0, n):
        for y in range(0, m):
            rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255, 255, 255) if (x+y) % 2 == 0 else (234, 234, 234), rect) 



    # N,nn*mm,75,25
def intervalFloat(i, maxim, upper=65, lower=25):
    y = 1.0- float((i)) / float((maxim))
    diff = int(str(upper))-int(str(lower))
    y = y*diff+int(str(lower))
    return y
    
def get_color(i, n):
    n = max(5,n)
    x = (1.0-float(i-1)/float(n-1))*0.5+0.3
    x=x*0.4+0.25
    r,g,b,a = matplotlib.colormaps['viridis'](x)
    return (r*255,g*255,b*255)

def genfood(snake):
    while 1:
        food = [random.randrange(0, n) +1 ,
            random.randrange(0, m) +1 ]
        if food not in snake:
            break
        if len(snake) >= n*m:
            food=[-2,-2]
            break
    return food

# Initialize game
def init_game():
    global snake, direction, food, steps, stepstotal, start_time, times
    snake = [[1,1]]
    #print("init")
    direction = 'RIGHT'
    food = genfood(snake)
    steps = [0]*(n*m)
    stepstotal = 0
    start_time = time.time()
    times = [0.0]*(n*m)


# Main game loop
pygame.joystick.init()
joysticks = []
for i in range(pygame.joystick.get_count()):
    joystick = pygame.joystick.Joystick(i)
    joystick.init()
    joysticks.append(joystick)

def main_game():
    global direction, steps, food, stepstotal, cell_size, width, height, times, snake
    toggle = False
    death = True
    # pygame.joystick.init()
    # if pygame.joystick.get_count() > 0:
    #     joystick = pygame.joystick.Joystick(0)
    #     joystick.init()
    #     #print(f"Detected joystick: {joystick.get_name()}")
    # else:
    #     print("No joystick detected")

    cell_size = min(width/n,height/m)
    load_svg(cell_size)
    snake=[[1,1]]
    #print("main")
    while len(snake)<n*m:
        key_pressed = False
        #resize = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != 'DOWN':
                    direction = 'UP'
                    key_pressed = True
                elif event.key == pygame.K_DOWN and direction != 'UP':
                    direction = 'DOWN'
                    key_pressed = True
                elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                    direction = 'LEFT'
                    key_pressed = True
                elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                    direction = 'RIGHT'
                    key_pressed = True
                if event.key == pygame.K_HASH:  # '#' key is pressed
                    #print("screenshot")
                    save_screenshot(screen)
                if event.key == pygame.K_t:  # 't' key is pressed
                    toggle = not toggle
                if event.key == pygame.K_a:  # 'a' key is pressed                    
                    food = genfood(snake)
                if event.key == pygame.K_d:  # 'a' key is pressed                    
                    death = not death
                if event.key == pygame.K_ESCAPE:  # 'esc' key is pressed                    
                    game_over()
                if event.key == pygame.K_MINUS:  # '-' key is pressed                    
                    snake.pop()
                if event.key == pygame.K_PLUS:  # '+' key is pressed    
                    if len(snake)>2:                
                        snake = snake[1:]
                #if event.key == pygame.K_LESS:  # '<' key is pressed                    
                #    width = 40*80                
                #    height = 32*80
            elif event.type == JOYAXISMOTION:

                if event.axis == 0:  # X-axis
                    if event.value < -0.95:
                        direction = 'LEFT'
                        key_pressed = True
                    elif event.value > 0.95:
                        direction = 'RIGHT'
                        key_pressed = True
                elif event.axis == 1:  # Y-axis
                    if event.value < -0.95:
                        direction = 'UP'
                        key_pressed = True
                    elif event.value > 0.95:
                        direction = 'DOWN'
                        key_pressed = True
            
            elif event.type == pygame.VIDEORESIZE:
                #global width, height
                width, height = event.size
                #screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                cell_size = min(width/n,height/m)
                load_svg(cell_size)
                #resize = True
            

        # Move the snake only if a key is pressed
        if key_pressed:
            steps[len(snake)] += 1
            stepstotal += 1


            head_x, head_y = snake[0]
            if direction == 'UP':
                head_y -= 1
            elif direction == 'DOWN':
                head_y += 1
            elif direction == 'LEFT':
                head_x -= 1
            elif direction == 'RIGHT':
                head_x += 1

            # Check for collision with walls
            
            wall = head_x <= 0 or head_x > n or head_y <= 0 or head_y > m
            #    game_over()
            carnivore = False
            # Check for collision with itself
            if [head_x, head_y] in snake:
                if [head_x, head_y] != snake[-1]:
                    if death:
                        game_over()
                    else:
                        carnivore = True

            # Move snake by adding the new head and removing the tail
            if not wall and not carnivore:
                snake.insert(0, [head_x, head_y])
                if snake[0] == food:
                    #print("StartFood", len(snake))
                    times[len(snake)-1] = time.time() - start_time
                    food = genfood(snake)
                    #visualize_stats(False)
                else:
                    if not toggle:
                        snake.pop()
        #print("DoneFood", len(snake))
        # Draw everything

        
        #if key_pressed or resize:
        
        draw_grid()
        alt = None
        for index, segment in enumerate(snake):
            radius = max(cell_size // 2, int(cell_size * (len(snake) - index) / len(snake)))  # Decrease radius towards the tail
            radius = intervalFloat(index,n*m)/80*cell_size
            color = get_color(index, len(snake))
            #print(color)
            if not alt == None:
                x = min(segment[0] , alt[0])
                y = min(segment[1] , alt[1])
                dx = abs(segment[0] - alt[0])
                if dx == 0:
                    pygame.draw.rect(screen, color, (x*cell_size-radius//2- cell_size // 2, y*cell_size-cell_size//2, 
                                             radius, cell_size), 0,3)
                else:
                    pygame.draw.rect(screen, color, (x*cell_size-cell_size//2, y*cell_size-radius//2 - cell_size // 2, 
                                             cell_size, radius), 0,3)
            alt = segment
        for index, segment in enumerate(snake):
            radius = max(cell_size // 2, int(cell_size * (len(snake) - index) / len(snake)))  # Decrease radius towards the tail
            radius = intervalFloat(index,n*m)/80*cell_size
            color = get_color(index, len(snake))
            pygame.draw.rect(screen, color, (segment[0]*cell_size-radius//2- cell_size // 2, segment[1]*cell_size-radius//2- cell_size // 2, 
                                             radius, radius), 0,3)
        draw_svg("eyes",((snake[0][0]-1)*cell_size,(snake[0][1]-1)*cell_size))
        #pygame.draw.circle(screen, red, (food[0] * cell_size - cell_size // 2, food[1] * cell_size - cell_size // 2), cell_size // 2)
        draw_svg("apple",((food[0]-1)*cell_size,(food[1]-1)*cell_size))

        # Draw score
        #score_text = font.render(f'Score: {score}', True, black)
        #screen.blit(score_text, [0, 0])

        #print("Draw")
        pygame.display.flip()
        clock.tick(fps)
        time.sleep(0.1)
    if len(snake) ==n*m:
        game_over()   

# Start menu
def start_menu():
    menu = pygame_menu.Menu('Welcome', width, height, theme=pygame_menu.themes.THEME_GREEN)

    menu.add.label(f"previous stats", align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(f"{len(snake)} steps, {int(sum(times))}s", align=pygame_menu.locals.ALIGN_LEFT)


    menu.add.button('Play', start_game)
    menu.add.text_input('Width: ', default=str(m), onchange=set_width)
    menu.add.text_input('Height: ', default=str(n), onchange=set_height)
    menu.add.button('Quit', pygame_menu.events.EXIT)
    while True:
            events = pygame.event.get()
            
            # Handle joystick events
            for event in events:
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Joystick button {event.button} pressed")
                    if event.button in [0, 1, 2, 3, 4, 5]:  # Buttons 0-3
                        # Perform the action of the currently selected widget
                        selected_widget = menu.get_selected_widget()
                        if isinstance(selected_widget, pygame_menu.widgets.Button):
                            selected_widget.apply()
                        elif isinstance(selected_widget, pygame_menu.widgets.TextInput):
                            # To simulate pressing Enter in a text input widget
                            selected_widget.apply()
                if event.type == pygame.JOYHATMOTION:
                    if event.value == (0, 1):  # Up
                        menu._select_previous_widget()
                    if event.value == (0, -1):  # Down
                        menu._select_next_widget()

            # Update the menu with events
            menu.update(events)
            menu.draw(screen)

            # Flip the display
            pygame.display.flip()
            time.sleep(0.1)

def set_height(value):
    global m
    if value =='':
        return
    if (int(value)>0):
        m = int(value)

def set_width(value):
    global n
    if value =='':
        return
    if (int(value)>0):
        n = int(value)

def start_game():
    global screen
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    init_game()
    main_game()

# Main
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
threading.Thread(target=display_plot, daemon=True).start()
start_menu()