import pygame
import math
import numpy as np
from evasion.envs.Agent import Agent

class GameDrawer:
    """
    helper class to help draw the game
    """
    def __init__(self) -> None:
        pass
    
    def draw_arrow(self, position, angle_deg):
        arrow_length = 20
        head_length = 5
        head_width = 10
        end = (position[0] + arrow_length * math.cos(math.radians(angle_deg)), 
               position[1] + arrow_length * math.sin(math.radians(angle_deg)))
        right_side = (end[0] + head_length * math.sin(math.radians(angle_deg)), 
                      end[1] - head_length * math.cos(math.radians(angle_deg)))
        left_side = (end[0] - head_length * math.sin(math.radians(angle_deg)), 
                     end[1] + head_length * math.cos(math.radians(angle_deg)))


        # pygame.draw.line(self.screen, (0, 0, 255), position, end, 5)
        # pygame.draw.polygon(self.screen, (0, 0, 255), [end, right_side, left_side])


        return [end, right_side, left_side]

    

# class GameRenderer:
#     def __init__(self,agent:Agent, width:int = 1000, height:int = 1000):
#         pygame.init()
#         self.width, self.height = width, height
#         self.screen = pygame.display.set_mode((self.width, self.height))
#         pygame.display.set_caption("Agent Visualization")
#         self.clock = pygame.time.Clock()
#         self.position = agent.current_state[:2]
#         self.angle = agent.current_state[2]
#         self.velocity = [0, 0]

#     def render(self, position, angle_rad):
#         self.handle_events()
#         self.draw(position, angle_rad)
#         self.clock.tick(60)

#     def handle_events(self):
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()

#     def draw_arrow(self, position, angle):
#         arrow_length = 20
#         head_length = 5
#         head_width = 10
#         end = (position[0] + arrow_length * math.cos(math.radians(angle)), 
#                position[1] + arrow_length * math.sin(math.radians(angle)))
#         right_side = (end[0] + head_length * math.sin(math.radians(angle)), 
#                       end[1] - head_length * math.cos(math.radians(angle)))
#         left_side = (end[0] - head_length * math.sin(math.radians(angle)), 
#                      end[1] + head_length * math.cos(math.radians(angle)))


#         pygame.draw.line(self.screen, (0, 0, 255), position, end, 5)
#         pygame.draw.polygon(self.screen, (0, 0, 255), [end, right_side, left_side])

#     def transform_position(self, position):
#         """Transform the position to lower left origin"""
#         x  = position[0]
#         y  = self.height - position[1]
#         return [x, y]

#     def draw(self, position, angle_rad):
#         self.screen.fill((255, 255, 255))
#         self.draw_arrow(position, angle_rad)
#         pygame.display.flip()
        
        
#     def manual_move(self) -> None:
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_UP]:
#             self.velocity[0] = speed * math.cos(math.radians(self.angle))
#             self.velocity[1] = speed * math.sin(math.radians(self.angle))
#         if keys[pygame.K_DOWN]:
#             self.velocity[0] *= 0.9
#             self.velocity[1] *= 0.9
#         if keys[pygame.K_LEFT]:
#             self.angle -= turn_speed
#         if keys[pygame.K_RIGHT]:
#             self.angle += turn_speed
            
#         self.position = [self.position[0] + self.velocity[0], 
#                          self.position[1] + self.velocity[1]]
#         self.render(self.position, self.angle)

# # Example usage in an OpenAI Gym environment
# if __name__ == "__main__":
#     renderer = GameRenderer()

#     # Example state
#     position = [10, 10]
#     angle = 0
#     # This would be inside your environment's step or render method
#     running = True
#     renderer.render(position, angle)
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
                
#         renderer.manual_move()
        

#     pygame.quit()