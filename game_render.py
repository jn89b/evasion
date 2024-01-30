import pygame
import math

from config import v_min, v_max, w_min, w_max, N_DISCRETE_ACTIONS, min_x, min_y, min_psi, max_x, max_y, max_psi, goal_x, goal_y, total_actions

import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Nonholonomic Vehicle Simulator")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Vehicle properties
position = [WIDTH // 2, HEIGHT // 2]
velocity = [0, 0]
angle = 0
speed = 5
turn_speed = 5

def draw_arrow(surface, color, position, angle):
    """ Draw an arrow representing the vehicle """
    arrow_length = 40
    head_length = 10
    head_width = 10
    end = (position[0] + arrow_length * math.cos(math.radians(angle)), position[1] + arrow_length * math.sin(math.radians(angle)))
    right_side = (end[0] + head_length * math.sin(math.radians(angle)), end[1] - head_length * math.cos(math.radians(angle)))
    left_side = (end[0] - head_length * math.sin(math.radians(angle)), end[1] + head_length * math.cos(math.radians(angle)))

    pygame.draw.line(surface, color, position, end, 5)
    pygame.draw.polygon(surface, color, [end, right_side, left_side])

# Game loop
running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        velocity[0] = speed * math.cos(math.radians(angle))
        velocity[1] = speed * math.sin(math.radians(angle))
    if keys[pygame.K_DOWN]:
        velocity[0] *= 0.9
        velocity[1] *= 0.9
    if keys[pygame.K_LEFT]:
        angle -= turn_speed
    if keys[pygame.K_RIGHT]:
        angle += turn_speed

    # Update position
    position[0] += velocity[0]
    position[1] += velocity[1]
    
    print(position)

    # Drawing
    screen.fill(WHITE)
    draw_arrow(screen, BLUE, position, angle)
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()
