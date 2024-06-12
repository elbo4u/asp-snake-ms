import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Arrow Keys and Gamepad Example')

# Initialize Joystick
pygame.joystick.init()
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected joystick: {joystick.get_name()}")
else:
    print("No joystick detected")

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_UP:
                print("Up arrow key pressed")
            elif event.key == K_DOWN:
                print("Down arrow key pressed")
            elif event.key == K_LEFT:
                print("Left arrow key pressed")
            elif event.key == K_RIGHT:
                print("Right arrow key pressed")
        elif event.type == KEYUP:
            if event.key == K_UP:
                print("Up arrow key released")
            elif event.key == K_DOWN:
                print("Down arrow key released")
            elif event.key == K_LEFT:
                print("Left arrow key released")
            elif event.key == K_RIGHT:
                print("Right arrow key released")
        elif event.type == JOYAXISMOTION:
            if event.axis == 0:  # X-axis
                if event.value < -0.1:
                    print("Left stick moved left")
                elif event.value > 0.1:
                    print("Left stick moved right")
            elif event.axis == 1:  # Y-axis
                if event.value < -0.1:
                    print("Left stick moved up")
                elif event.value > 0.1:
                    print("Left stick moved down")
        elif event.type == JOYBUTTONDOWN:
            print(f"Joystick button {event.button} pressed")
        elif event.type == JOYBUTTONUP:
            print(f"Joystick button {event.button} released")

pygame.quit()
