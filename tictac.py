import numpy as np
import random

# Initialize board and game parameters
board = np.array([['-', '-', '-'],
                  ['-', '-', '-'],
                  ['-', '-', '-']])

players = ['x', '0']  # 'x' for human player, '0' for agent
num_players = len(players)

Q = {}
learning_rate = 0.001
discount_factor = 0.9
exploration_rate = 0.5
num_episodes = 10000
num_games = 1000  # Number of games for evaluation

# Functions for the game
def print_board(board):
    for row in board:
        print(' | '.join(row))
    print()

def board_to_string(board):
    return ''.join(board.flatten())

def is_game_over(board):
    for row in board:
        if len(set(row)) == 1 and row[0] != '-':
            return True, row[0]
    
    for col in board.T:
        if len(set(col)) == 1 and col[0] != '-':
            return True, col[0]
    
    if len(set(board.diagonal())) == 1 and board[0, 0] != '-':
        return True, board[0, 0]
    
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0, 2] != '-':
        return True, board[0, 2]
    
    if '-' not in board:
        return True, 'draw'
    
    return False, None

def board_next_state(board, cell, player):
    next_state = board.copy()
    next_state[cell[0], cell[1]] = player
    return next_state

def update_q_table(state, action, next_state, reward):
    q_values = Q.get(state, np.zeros((3, 3)))
    next_q_values = Q.get(board_to_string(next_state), np.zeros((3, 3)))
    max_next_q_value = np.max(next_q_values)
    
    q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])
    Q[state] = q_values

def choose_action(board, exploration_rate):
    if random.uniform(0, 1) < exploration_rate:
        available_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == '-']
        return random.choice(available_moves)
    else:
        state = board_to_string(board)
        q_values = Q.get(state, np.zeros((3, 3)))
        available_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == '-']
        best_move = max(available_moves, key=lambda move: q_values[move[0], move[1]])
        return best_move

# Training the agent
num_draws = 0
agent_wins = 0

for episode in range(num_episodes):
    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])
    
    current_player = random.choice(players)
    game_over = False

    while not game_over:
        action = choose_action(board, exploration_rate)
        row, col = action
        board[row, col] = current_player

        game_over, winner = is_game_over(board)
        
        if game_over:
            if winner == current_player:
                reward = 1
                if current_player == '0':
                    agent_wins += 1
            elif winner == 'draw':
                reward = 0.5
                num_draws += 1
            else:
                reward = 0
            update_q_table(board_to_string(board), action, board, reward)
        else:
            next_state = board_next_state(board, action, players[(players.index(current_player) + 1) % num_players])
            update_q_table(board_to_string(board), action, next_state, 0)
            current_player = players[(players.index(current_player) + 1) % num_players]
    
    exploration_rate *= 0.99

agent_win_percentage = (agent_wins / num_episodes) * 100
draw_percentage = (num_draws / num_episodes) * 100

print("Agent win percentage: {:.2f}%".format(agent_win_percentage))
print("Draw percentage: {:.2f}%".format(draw_percentage))

# Playing against the trained agent
def play_game():
    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])
    
    current_player = random.choice(players)
    game_over = False

    while not game_over:
        if current_player == 'x':  # Human player
            print_board(board)
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column (0-2): "))
            action = (row, col)
        else:  # Agent
            action = choose_action(board, exploration_rate=0)
        
        row, col = action
        board[row, col] = current_player
        game_over, winner = is_game_over(board)
        
        if game_over:
            print_board(board)
            if winner == 'x':
                print("Human player wins!")
            elif winner == '0':
                print("Agent wins!")
            else:
                print("It's a draw!")
        else:
            current_player = players[(players.index(current_player) + 1) % num_players]

play_game()
