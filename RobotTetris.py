from collections import OrderedDict
import random

from pygame import Rect
import pygame
import numpy as np
import queue
from tqdm import tqdm
from statistics import mean
from Agent import DQNAgent
from logs import CustomTensorBoard
from datetime import datetime
from pygame.locals import *

import os
#import psutil

#def show_RAM_usage():
#    py = psutil.Process(os.getpid())
#    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))


from keras.models import load_model


WINDOW_WIDTH, WINDOW_HEIGHT = 500, 601
GRID_WIDTH, GRID_HEIGHT = 300, 600
TILE_SIZE = 30
BOARD_WIDTH = int(GRID_WIDTH/TILE_SIZE)
BOARD_HEIGHT = int(GRID_HEIGHT/TILE_SIZE)
LIGHTBLUE1 = (175,238,238) #light blue  
YELLOW1 = (228,253,31) #yellow
DARKBLUE1 = (0,109,176) #dark blue
ORANGE1 = (255,165,0) #orange
GREEN1 = (143,206,0) #green
RED1 = (255,0,0) #red
PINK1 = (255,192,203) #pink
BLACK = (0,0,0)
WHITE = (255,255,255)



def remove_empty_columns(arr, _x_offset=0, _keep_counting=True):
    """
    Remove empty columns from arr (i.e., those filled with zeros).
    The return value is (new_arr, x_offset), where x_offset is how
    much the x coordinate needs to be increased in order to maintain
    the block's original position.
    """
    for colid, col in enumerate(arr.T):
        if col.max() == 0:
            if _keep_counting:
                _x_offset += 1
            # Remove the current column and try again.
            arr, _x_offset = remove_empty_columns(
                np.delete(arr, colid, 1), _x_offset, _keep_counting)
            break
        else:
            _keep_counting = False
    return arr, _x_offset


class BottomReached(Exception):
    pass


class TopReached(Exception):
    pass


class Block(pygame.sprite.Sprite):

    block_id = 0

    @staticmethod
    def collide(block, group):
        """
        Check if the specified block collides with some other block
        in the group.
        """
        for other_block in group:
            # Ignore the current block which will always collide with itself.
            if block == other_block:
                continue
            if pygame.sprite.collide_mask(block, other_block) is not None:
                return True
        return False

    def __init__(self):
        super().__init__()
        # Get a random color.
        #self.color = random.choice((
        #   (175,238,238), #light blue
        #    (0,109,176), #dark blue
        #    (255,150,102), #orange
        #    (228,253,31), #yellow
        #    (211,255,206), #green
        #    (255,192,203), #pink
        #    (255,0,0), #red
        #))
        Block.block_id += 1
        self.id = Block.block_id
        self.current = True
        self.struct = np.array(self.struct)
        self._draw()

    def _draw(self, x=4, y=0):
        width = len(self.struct[0]) * TILE_SIZE
        height = len(self.struct) * TILE_SIZE
        self.image = pygame.surface.Surface([width, height])
        self.image.set_colorkey((0, 0, 0))
        # Position and size
        self.rect = Rect(0, 0, width, height)
        self.x = x
        self.y = y
        for y, row in enumerate(self.struct):
            for x, col in enumerate(row):
                if col:
                    pygame.draw.rect(
                        self.image,
                        self.color,
                        Rect(x*TILE_SIZE + 1, y*TILE_SIZE + 1,
                            TILE_SIZE - 2, TILE_SIZE - 2)
                    )
        self._create_mask()

    def redraw(self):
        self._draw(self.x, self.y)

    def _create_mask(self):
        """
        Create the mask attribute from the main surface.
        The mask is required to check collisions. This should be called
        after the surface is created or update.
        """
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def group(self):
        return self.groups()[0]

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.rect.left = value*TILE_SIZE

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.rect.top = value*TILE_SIZE

    def move_left(self, group):
        self.x -= 1
        # Check if we reached the left margin.
        if self.x < 0 or Block.collide(self, group):
            self.x += 1

    def move_right(self, group):
        self.x += 1
        # Check if we reached the right margin or collided with another
        # block.
        if self.rect.right > GRID_WIDTH or Block.collide(self, group):
            # Rollback.
            self.x -= 1

    def move_down_auto(self, group):
        self.y += 1
        # Check if the block reached the bottom or collided with
        # another one.
        if self.rect.bottom > GRID_HEIGHT or Block.collide(self, group):
            # Rollback to the previous position.
            self.y -= 1
            self.current = False
            raise BottomReached
    def move_down_press(self, group):
        while(1):
            self.y += 1
            # Check if the block reached the bottom or collided with
            # another one.
            if self.rect.bottom > GRID_HEIGHT or Block.collide(self, group):
                # Rollback to the previous position.
                self.y -= 1
                self.current = False
                raise BottomReached


    def rotate(self, group):
        self.image = pygame.transform.rotate(self.image, -90) #-90 because we want to rotate to the right
        # Once rotated we need to update the size and position.
        self.rect.width = self.image.get_width()
        self.rect.height = self.image.get_height()
        self._create_mask()
        # Check the new position doesn't exceed the limits or collide
        # with other blocks and adjust it if necessary.
        while self.rect.right > GRID_WIDTH:
            self.x -= 1
        while self.rect.left < 0:
            self.x += 1
        while self.rect.bottom > GRID_HEIGHT:
            self.y -= 1
        while True:
            if not Block.collide(self, group):
                break
            self.y -= 1
        self.struct = np.rot90(self.struct, -1)

    def update(self):
        if self.current:
            self.move_down_auto()


class IBlock(Block):
    struct = (
        (1,1,1,1),
    )
    def __init__(self):
        self.color = LIGHTBLUE1
        self.type = IBlock
        super().__init__()


    

class OBlock(Block):
    struct = (
        (1, 1),
        (1, 1)
    )
    def __init__(self):
        self.color = YELLOW1 
        self.type = OBlock
        super().__init__()

class JBlock(Block):
    struct = (
        (1, 0, 0),
        (1, 1, 1)
    )
    def __init__(self):
        self.color = DARKBLUE1
        self.type = JBlock
        super().__init__()

class LBlock(Block):
    struct = (
        (0, 0, 1),
        (1, 1, 1)
    )
    def __init__(self):
        self.color = ORANGE1
        self.type = LBlock
        super().__init__()

class SBlock(Block):
    struct = (
        (0, 1, 1),
        (1, 1, 0)
    )
    def __init__(self):
        self.color = GREEN1
        self.type = SBlock
        super().__init__()

class ZBlock(Block):
    struct = (
        (1, 1, 0),
        (0, 1, 1)
    )
    def __init__(self):
        self.color = RED1
        self.type = ZBlock
        super().__init__()

class TBlock(Block):
    struct = (
        (0, 1, 0),
        (1, 1, 1)
    )
    def __init__(self):
        self.color = PINK1
        self.type = TBlock
        super().__init__()



class TETRIS(pygame.sprite.OrderedUpdates):

    """
    The game of TETRIS is treated as a collection of all blocks that currently exist.
    """
    @staticmethod
    def get_random_block():
        return random.choice(
            (IBlock, OBlock, JBlock, LBlock, SBlock, ZBlock, TBlock))()
    
    @staticmethod
    def get_random_block_permutation():
        blocks = [IBlock, OBlock, JBlock, LBlock, SBlock, ZBlock, TBlock]
        random.shuffle(blocks)
        queue_blocks = queue.Queue()
        for block in blocks:
            queue_blocks.put(block())
        return queue_blocks

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.reset()

        

    def reset(self):
        self._reset_grid()
        for block in self:
            self.remove(block)
        self._ignore_next_stop = False
        self.score = 0
        self.next_block = None
        # Not really moving, just to initialize the attribute.
        self.stop_moving_current_block()
        # Create first Permutation
        self.block_bag = TETRIS.get_random_block_permutation()
        # The first block.
        self._pick_new_block()


    def _check_line_completion(self, completed_lines = 0):
        """
        Check each line of the grid and remove the ones that
        are complete.
        """
        # Start checking from the bottom.
        for i, row in enumerate(self.grid[::-1]):
            if all(row):
                self.score += 10
                # Get the blocks affected by the line deletion and
                # remove duplicates.
                affected_blocks = list(
                    OrderedDict.fromkeys(self.grid[-1 - i]))
                #print(affected_blocks)

                for block, y_offset in affected_blocks:
                    #print(block, y_offset)
                    # Remove the block tiles which belong to the
                    # completed line.
                    block.struct = np.delete(block.struct, y_offset, 0)
                    if block.struct.any():
                        # Once removed, check if we have empty columns
                        # since they need to be dropped.
                        block.struct, x_offset = \
                            remove_empty_columns(block.struct)
                        # Compensate the space gone with the columns to
                        # keep the block's original position.
                        block.x += x_offset
                        # Force update.
                        block.redraw()
                    else:
                        # If the struct is empty then the block is gone.
                        self.remove(block)



                #Move every block that is above the removed line down

                for block in self:
                    #print(block, " keks", block.id)

                    if block.y < BOARD_HEIGHT - i:
                        try:
                            block.move_down_auto(self)
                        except BottomReached:
                            continue
                    #else:
                        #print(block, " nisko sam ", block.id, "y and i are", block.y , i)

                self.update_grid()
                # Since we've updated the grid, now the i counter
                # is no longer valid, so call the function again
                # to check if there're other completed lines in the
                # new grid.
                completed_lines += 1
                return self._check_line_completion(completed_lines)
        #if we didnt find any completed lines
        return 0

    def _reset_grid(self):
        self.grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

    def _pick_new_block(self):
        new_block = self.next_block or self.block_bag.get()
        if Block.collide(new_block, self):
            raise TopReached
        self.add(new_block)
        if(self.block_bag.empty()):
            self.block_bag = TETRIS.get_random_block_permutation()
        self.next_block = self.block_bag.get()
        self.update_grid()

    def update_grid(self):
        self._reset_grid()
        for block in self:
            for y_offset, row in enumerate(block.struct):
                for x_offset, digit in enumerate(row):
                    # Prevent replacing previous blocks.
                    if digit == 0:
                        continue
                    rowid = block.y + y_offset
                    colid = block.x + x_offset
                    self.grid[rowid][colid] = (block, y_offset)

    @property
    def current_block(self):
        return self.sprites()[-1]


    def start_moving_current_block(self, key):
        if self._current_block_movement_heading is not None:
            self._ignore_next_stop = True
        self._current_block_movement_heading = key

    def stop_moving_current_block(self):
        if self._ignore_next_stop:
            self._ignore_next_stop = False
        else:
            self._current_block_movement_heading = None

    def rotate_current_block(self):
        # Prevent SquareBlocks rotation.
        if not isinstance(self.current_block, OBlock):
            self.current_block.rotate(self)
            self.update_grid()

    def get_score(self):
        return self.score

    ##Functions specific for AI

    def play_move(self, x, rotation):

        # First check if there's something to move.
        while(rotation > 0):
            self.current_block.rotate(self)
            rotation -=1
        self.current_block.x = x

        game_over = False
        completed_lines = 0
        try:
            self.current_block.move_down_press(self)
        except BottomReached:
            self.stop_moving_current_block()
            #Check here for game over (when there is anything in the top 2 rows)
            self.update_grid()
            completed_lines = self._check_line_completion()
            if any(self.grid[0]) or any(self.grid[1]):
                game_over = True

            try:
                self._pick_new_block()
            except TopReached:
                game_over = True
        else:
            self.update_grid()

        reward = 1 + (completed_lines ** 2) * BOARD_WIDTH

        if game_over == True:
            reward -= 2

        return reward, game_over
    
    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        current_block_type = self.current_block.type
        
        if current_block_type == OBlock: 
            rotations = [0]
        elif current_block_type == IBlock:
            rotations = [0, 1]
        else:
            rotations = [0, 1, 2, 3]
        starting_x = self.current_block.x
        starting_y = self.current_block.y
        # For all rotations
        for rotation in rotations:
            self.current_block.x = starting_x
            rotation_count = rotation
            while(rotation_count > 0):
                self.current_block.rotate(self)
                rotation_count -=1

            for i in range(0,6):
                self.current_block.move_left(self)
            min_x = self.current_block.x

            for i in range(0,10):
                self.current_block.move_right(self)
            max_x = self.current_block.x
            #print("max_x =", max_x)


            # For all positions
            for x in range(min_x, max_x+1):
                #calculate the resulting board
                board = self.calculate_board_if_dropped(x, starting_y)
                #Save the "states" from the calculated board
                states[(x, rotation)] = self.get_board_state_properties(board)
                #return the blocks to their original positions
                self.current_block.x = starting_x
                self.current_block.y = starting_y
                self.update_grid()
            
            #rotate the block back to starting rotation
            if(rotation):
                while(4 - rotation > 0):
                    self.current_block.rotate(self)
                    rotation +=1


        return states
    
    def calculate_board_if_dropped(self, x, y):
        self.current_block.x = x
        self.current_block.y = y

        try:
            self.current_block.move_down_press(self)
        except BottomReached:
            self.stop_moving_current_block()
            self.update_grid()
            board = [[True if val != 0 else False for val in row] for row in self.grid]
            #print(board)
            return board
        
        return None

    def get_board_state_properties(self, board):
        '''Get properties of the board'''
        full_lines = self.full_lines_count(board)
        holes = self.hole_count(board)
        bumpiness = self.bumpiness_count(board)
        sum_height = self.sum_of_heights(board)
        return [full_lines, holes, bumpiness, sum_height]
    
    def full_lines_count(self, board):
        count = 0
        for row in board:
            if all(row):
                count += 1
        return count
    
    def hole_count(self, board):
        count = 0
        for col in range(len(board[0])):
            for row in range(1, len(board)):
                if not board[row][col] and board[row-1][col]:
                    count += 1
        return count
    
    def bumpiness_count(self, board):
        heights = []
        for col in range(len(board[0])):
            for row in range(len(board)):
                if board[row][col]:
                    heights.append(len(board) - row)
                    break
            else:
                heights.append(0)
        diffs = [abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)]
        return sum(diffs)

    def sum_of_heights(self, board):
        heights = []
        for col in range(len(board[0])):
            for row in range(len(board)):
                if board[row][col]:
                    heights.append(len(board) - row)
                    break
            else:
                heights.append(0)
        return sum(heights)

    

def draw_grid(background):
    """Draw the background grid."""
    grid_color = 50, 50, 50
    # Vertical lines.
    for i in range(11):
        x = TILE_SIZE * i
        pygame.draw.line(
            background, grid_color, (x, 0), (x, GRID_HEIGHT)
        )
    # Horizontal liens.
    for i in range(21):
        y = TILE_SIZE * i
        pygame.draw.line(
            background, grid_color, (0, y), (GRID_WIDTH, y)
        )


def draw_centered_surface(screen, surface, y):
    screen.blit(surface, (400 - surface.get_width()//2, y))



def display_message(screen, font, message, pos):
    text = font.render(message, True, (255, 255, 255))
    rect = text.get_rect(center=pos)
    screen.blit(text, rect)

def ask_training_mode(screen, font):
    screen.fill((0, 0, 0))
    display_message(screen, font, "Want to train, or let the bot play?", (screen.get_width() // 2, 100))
    train_button = pygame.draw.rect(screen, (0, 255, 0), (50, 200, 150, 50))
    play_button = pygame.draw.rect(screen, (255, 0, 0), (250, 200, 150, 50))
    display_message(screen, font, "Train", train_button.center)
    display_message(screen, font, "Play", play_button.center)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()
            elif event.type == MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if train_button.collidepoint(pos):
                    return True
                elif play_button.collidepoint(pos):
                    return False

def main():
    #ask the player if he wants to train, or let the AI play
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Tetris")
    font = pygame.font.SysFont('arial', 20)
    Training = ask_training_mode(screen, font)
    pygame.quit()


    pygame.init()
    pygame.display.set_caption("Tetris")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    game_over = False
    # Create background.
    background = pygame.Surface(screen.get_size())
    bgcolor = BLACK
    background.fill(bgcolor)
    # Draw the grid on top of the background.
    draw_grid(background)
    # This makes blitting faster.
    background = background.convert()
    font = pygame.font.SysFont('arial', 20)

    next_block_text = font.render("Next figure:", True, WHITE, bgcolor)
    score_msg_text = font.render("Score:", True, WHITE, bgcolor)
    game_over_text = font.render("Game over!", True, YELLOW1, bgcolor)

    
    EVENT_MOVE_CURRENT_BLOCK = pygame.USEREVENT + 2
    pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, 100)
    tetris = TETRIS()

    #The training process and gameplay
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1200
    mem_size = 20000
    discount = 0.95
    batch_size = 512 #512
    epochs = 1
    render_every = 25
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32] #[32, 32] 
    render_delay = None
    save_model_every = 25
    activations = ['relu', 'relu', 'linear'] #['relu', 'relu', 'linear']
    state_size = 4

    #here you give the path and the name to an already existing model that you created
    model_dir = f'models/tetris-episode=1650'
    model_path = os.path.join(model_dir, 'current_model.h5')
    replay_path = os.path.join(model_dir, 'replay_buffer.bin')
    agent = DQNAgent(state_size,
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size,
                     pretrained_model=not Training, model_file=model_path, continueing=False, buffer_file = replay_path)
    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    if Training == True:
        for episode in tqdm(range(episodes)):
            #print("test0")
            #Where to save model
            model_dir = f'models/tetris-episode={episode}'
            current_state = (0,0,0,0)
            tetris.reset()
            steps = 0
            game_over = False

            if render_every and episode % render_every == 0:
                render = True
            else:
                render = False
           
            #Game
            while not game_over:
                # Event handling
                for event in pygame.event.get():
                    # User input collection
                    if event.type == pygame.QUIT:
                        game_over = True
                        break
                    

                    # Movement Handling
    
                
                # AI move
                next_states = tetris.get_next_states()
                best_state = agent.best_state(next_states.values())

                best_action = None #this is an array [x, rotation]
                for action, state in next_states.items():
                    if state == best_state:
                        best_action = action
                        break

                reward, game_over = tetris.play_move(best_action[0], best_action[1])


                agent.add_to_memory(current_state, next_states[best_action], reward, game_over)

                #print(current_state, next_states[best_action], reward, game_over)
                current_state = next_states[best_action]
                steps += 1


                #--------------------------------------------------
                if render == True:
                    # Draw background and grid.
                    screen.blit(background, (0, 0))
                    # Blocks.
                    tetris.draw(screen)
                    # Sidebar with misc. information.
                    draw_centered_surface(screen, next_block_text, 50)
                    draw_centered_surface(screen, tetris.next_block.image, 100)
                    draw_centered_surface(screen, score_msg_text, 240)
                    score_text = font.render(
                        str(tetris.score), True, WHITE, bgcolor)
                    draw_centered_surface(screen, score_text, 270)
                    if game_over:
                        draw_centered_surface(screen, game_over_text, 360)
                        print("Final Score is",tetris.score)
                    # Update.
                    pygame.display.flip()

                
            
            scores.append(tetris.get_score())
            #print("test1")
            # Train
            if episode % train_every == 0:
                #show_RAM_usage()
                agent.train(batch_size=batch_size, epochs=epochs)


            if (episode % save_model_every == 0) and (episode > 300):
                
                agent.save_model_to_dir(model_dir)
                replay_path = os.path.join(model_dir, 'replay_buffer.bin')
                agent.save_replay_buffer(replay_path)

            #print("test2")
            # Logs
            if log_every and episode and episode % log_every == 0:
                avg_score = mean(scores[-log_every:])
                min_score = min(scores[-log_every:])
                max_score = max(scores[-log_every:])

                log.log(episode, avg_score=avg_score, min_score=min_score,
                        max_score=max_score)
            #print("test4")
    else:
        while(1):
            current_state = (0,0,0,0)
            tetris.reset()
            game_over = False
            render=True

            
            #Game
            while not game_over:
                # Event handling
                for event in pygame.event.get():
                    # User input collection
                    if event.type == pygame.QUIT:
                        game_over = True
                        break
                    

                # Movement Handling
    
                # AI move
                next_states = tetris.get_next_states()
                best_state = agent.best_state(next_states.values())

                best_action = None #this is an array [x, rotation]
                for action, state in next_states.items():
                    if state == best_state:
                        best_action = action
                        break

                reward, game_over = tetris.play_move(best_action[0], best_action[1])

                #--------------------------------------------------
                if render == True:
                    # Draw background and grid.
                    screen.blit(background, (0, 0))
                    # Blocks.
                    tetris.draw(screen)
                    # Sidebar with misc. information.
                    draw_centered_surface(screen, next_block_text, 50)
                    draw_centered_surface(screen, tetris.next_block.image, 100)
                    draw_centered_surface(screen, score_msg_text, 240)
                    score_text = font.render(
                        str(tetris.score), True, WHITE, bgcolor)
                    draw_centered_surface(screen, score_text, 270)
                    if game_over:
                        draw_centered_surface(screen, game_over_text, 360)
                        print("Final Score is",tetris.score)
                    # Update.
                    pygame.display.flip()


    pygame.quit()


if __name__ == "__main__":
    main()

    