import pygame
from engine import Board, GameWindow

width, height = 600, 700
thickness = 50

win = pygame.display.set_mode((width, height))
pygame.display.set_caption('2048')

game = GameWindow(4, width, height, thickness)

def redraw_window(win):
    game.draw(win)
    pygame.display.update()

def main():
    
    clock = pygame.time.Clock()
 
    # Main loop
    while True:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # This enables keyboard interaction with arrow keys
        # Comment it out if you want to test some other form of input
        game.move()

        redraw_window(win)

        if (game.board.check_loss()):
            response = None
            while response not in ['Y', 'N']:
                print(game.board)
                response = input('You lost! Restart? (y/n) ').upper()
                if response.upper() == 'Y':
                    game.board.restart()
                else:
                    pygame.quit()

        # ADD NEW INPUT CONDITIONS HERE
        # Change these conditions to whatever input you would like to test with
        # if False:
        #     b.move_left()
        # if False:
        #     b.move_right()
        # if False:
        #     b.move_up()
        # if False:
        #     b.move_down()

main()