import copy
import random
import pygame
    
class Board():
    def __init__(self, size = 4, score = 0, array = []):
        self.size = size
        self.array = array
        if array == []:
            for i in range(size):
                self.array.append([0] * size)
        self.score = score

    def copy(self):
        return Board(self.size, self.score, copy.deepcopy(self.array))

    def add_tile(self):
        # Adds a tile to the game board (2 with 90% chance and 4 with 10% chance)
        open_spots = []
        for x in range(self.size):
            for y in range(self.size):
                if self.array[x][y] == 0:
                    open_spots.append([x, y])
        if not open_spots:
            return
        new_spot = random.choice(open_spots)
        x, y = new_spot[0], new_spot[1]
        self.array[x][y] = random.choices([2, 4], weights = [0.9, 0.1])[0]
    
    def restart(self):
        # Create a fresh board and place two starting tiles
        self.score = 0
        self.array = []
        for i in range(self.size):
            self.array.append([0] * self.size)
        self.add_tile()
        self.add_tile()

        return self.array, self.valid_moves()

    def board_full(self):
        # Check if there are any open spots on the board
        for row in self.array:
            if 0 in row:
                return False
        return True
    
    def check_loss(self):
        # We can only lose if every tile is filled so check that first
        if not self.board_full():
            return False

        # Check if it is possible to move in any of the four directions
        return not self.can_move_left() and not self.can_move_right() and not self.can_move_up() and not self.can_move_down()

    def can_move_left(self):
        temp_board = Board(size = self.size, array = copy.deepcopy(self.array))
        temp_board.move_left(add_tile=False)
        return temp_board.array != self.array

    def can_move_right(self):
        temp_board = Board(size = self.size, array = copy.deepcopy(self.array))
        temp_board.move_right(add_tile=False)
        return temp_board.array != self.array

    def can_move_up(self):
        temp_board = Board(size = self.size, array = copy.deepcopy(self.array))
        temp_board.move_up(add_tile=False)
        return temp_board.array != self.array
    
    def can_move_down(self):
        temp_board = Board(size = self.size, array = copy.deepcopy(self.array))
        temp_board.move_down(add_tile=False)
        return temp_board.array != self.array

    def move_left(self, add_tile = True):
        # Move the board left, combining tiles as they move
        for row in self.array:
            for x in range(self.size):
                last_tile = 0
                for i in range(x-1, last_tile-1, -1):
                    if row[i] == 0:
                        if i == last_tile:
                            row[i] = int(row[x])
                            row[x] = 0
                            break
                    elif row[i] == row[x]:
                        row[i] *= 2
                        self.score += row[i]
                        row[x] = 0
                        last_tile = i + 1
                        break
                    else:
                        row[i+1] = int(row[x])
                        if i + 1 != x:
                            row[x] = 0
                        break
        if add_tile:
            self.add_tile()

    def move_right(self, add_tile = True):
        # Move the board right, combining tiles as they move
        for row in self.array:
            row.reverse()
            for x in range(self.size):
                last_tile = 0
                for i in range(x-1, last_tile-1, -1):
                    if row[i] == 0:
                        if i == last_tile:
                            row[i] = int(row[x])
                            row[x] = 0
                            break
                    elif row[i] == row[x]:
                        row[i] *= 2
                        self.score += row[i]
                        row[x] = 0
                        last_tile = i + 1
                        break
                    else:
                        row[i+1] = int(row[x])
                        if i + 1 != x:
                            row[x] = 0
                        break
            row.reverse()
        if add_tile:
            self.add_tile()

    def move_up(self, add_tile = True):
        # Move the board up, combining tiles as they move
        for y in range(self.size):
            col = []
            for x in range(self.size):
                col.append(int(self.array[x][y]))
            for x in range(self.size):
                last_tile = 0
                for i in range(x-1, last_tile-1, -1):
                    if col[i] == 0:
                        if i == last_tile:
                            col[i] = int(col[x])
                            col[x] = 0
                            break
                    elif col[i] == col[x]:
                        col[i] *= 2
                        self.score += col[i]
                        col[x] = 0
                        last_tile = i + 1
                        break
                    else:
                        col[i+1] = int(col[x])
                        if i + 1 != x:
                            col[x] = 0
                        break
            for x in range(self.size):
                self.array[x][y] = col[x]
        if add_tile:
            self.add_tile()

    def move_down(self, add_tile = True):
        # Move the board down, combining tiles as they move
        for y in range(self.size):
            col = []
            for x in range(self.size):
                col.append(int(self.array[x][y]))
            col.reverse()
            for x in range(self.size):
                last_tile = 0
                for i in range(x-1, last_tile-1, -1):
                    if col[i] == 0:
                        if i == last_tile:
                            col[i] = int(col[x])
                            col[x] = 0
                            break
                    elif col[i] == col[x]:
                        col[i] *= 2
                        self.score += col[i]
                        col[x] = 0
                        last_tile = i + 1
                        break
                    else:
                        col[i+1] = int(col[x])
                        if i + 1 != x:
                            col[x] = 0
                        break
            col.reverse()
            for x in range(self.size):
                self.array[x][y] = col[x]
        if add_tile:
            self.add_tile()

    def valid_moves(self):
        valid_moves = []
        if self.can_move_up():
            valid_moves.append(0)
        if self.can_move_right():
            valid_moves.append(1)
        if self.can_move_down():
            valid_moves.append(2)
        if self.can_move_left():
            valid_moves.append(3)
        return valid_moves

    def step(self, action):
        old_score = self.score

        if action == 0:
            self.move_up()
        if action == 1:
            self.move_right()
        if action == 2:
            self.move_down()
        if action == 3:
            self.move_left()

        reward = self.score - old_score

        return self.array, reward, self.check_loss(), self.valid_moves()

    def __repr__(self):
        return "\n".join([str(row) for row in self.array])

class GameWindow:
    def __init__(self, n = 4, width = 600, height = 600, thickness = 25, font = None):
        self.width = width
        self.height = height
        self.thickness = thickness
        self.rect = (width / thickness, height - width * ((thickness - 1) / thickness), width * ((thickness - 2) / thickness), width * ((thickness - 2) / thickness))
        self.font = font
        if self.font == None:
            pygame.font.init()
            self.font = pygame.font.SysFont('Comic Sans MS', 30)

        self.board = Board(size=n)
        self.board.restart()

    def draw(self, win):
        win.fill((255, 255, 255))

        # Score
        text = self.font.render('Score : ' + str(self.board.score), True, (0,0,0))
        textRect = text.get_rect()
        textRect.center = (self.width / 2, 50)
        win.blit(text, textRect)

        # Main Board Space
        pygame.draw.rect(win, (160, 160, 160), self.rect)

        # Tiles
        length = (self.thickness - 2 - self.board.size - 1) * self.width / (self.thickness * self.board.size)
        for x in range(self.board.size):
            for y in range(self.board.size):
                rect = (self.rect[0] + (self.width * (x + 1) / self.thickness) + length * x, self.rect[1] + (self.width * (y + 1) / self.thickness) + length * y, length, length)
                pygame.draw.rect(win, self.color(self.board.array[y][x]), rect)
                # Tile Value
                if self.board.array[y][x] != 0:
                    text = self.font.render(str(self.board.array[y][x]), True, (0,0,0))
                    textRect = text.get_rect()
                    textRect.center = (rect[0] + length / 2, rect[1] + length / 2)
                    win.blit(text, textRect)

    def move(self):
        # Move the board based on the current key being pressed
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            if self.board.can_move_left():
                self.board.move_left()
        
        elif keys[pygame.K_RIGHT]:
            if self.board.can_move_right():
                self.board.move_right()

        elif keys[pygame.K_UP]:
            if self.board.can_move_up():
                self.board.move_up()

        elif keys[pygame.K_DOWN]:
            if self.board.can_move_down():
                self.board.move_down()

    def color(self, val):
        colors = {
            0: (180, 180, 180),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 197, 63),
            1024: (237, 197, 63),
            2048: (237, 197, 30),
            4096: (100, 184, 145),
            8192: (56, 140, 100),
            16384: (56, 107, 126),
        }
        return colors[val]

