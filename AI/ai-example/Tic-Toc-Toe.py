
# import math library for calculation
import math

# evaluate if there are three symbols in a line vertically, horizontally, or diagonally
def evaluate(board):
    winner = None   # declare a string for determining whether there is a winner
    
    # loops
    for player in ['X', 'O']:
        if any(all(board[i] == player for i in row) for row in [
            range(0, 9, 3), range(1, 9, 3), range(2, 9, 3),  # rows
            range(0, 7, 3), range(1, 8, 3), range(2, 9, 3),  # columns
            range(0, 9, 4), range(2, 7, 2),  # diagonals
        ]):
            winner = player
            break

    # return 1 for default winner 'O', -1 for default winner 'X'
    if winner == 'X':
        return -1
    elif winner == 'O':
        return 1
    elif '-' not in board:
        return 0
    else:
        return None

# minimax algorithm
def minimax(board, maximizing_player):

    # first evaluating the if there is a winner on the board
    result = evaluate(board)
    # if there is a winner (result has value), return the winner
    if result is not None:
        return result

    # record the current best score
    best_score = -math.inf if maximizing_player else math.inf

    # dive deeper to the adversasial tree
    for i, cell in enumerate(board):
        if cell == '-':
            board[i] = 'O' if maximizing_player else 'X'
            score = minimax(board, not maximizing_player)
            board[i] = '-'
            if maximizing_player:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

    return best_score

# randomly get the computer's move and add it to the game board
def get_computer_move(board):
    best_score = -math.inf
    best_move = None
    for i, cell in enumerate(board):
        if cell == '-':
            board[i] = 'O'
            # the computer will choose the minimize its outcome value for the next move
            score = minimax(board, False)
            board[i] = '-'
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

# print our the game board one row by one row
def print_board(board):
    print(board[0:3])
    print(board[3:6])
    print(board[6:9])
    print()

def play_game():
    print("game board")
    board = ['-' for i in range(9)]
    print_board(board)
    while True:
        human_move = int(input("Enter your move (0-8): "))
        if board[human_move] == '-':
            board[human_move] = 'X'
            print_board(board)
            if evaluate(board) is not None:
                print("Game over!")
                break
            computer_move = get_computer_move(board)
            board[computer_move] = 'O'
            print("Computer's move")
            print_board(board)
            if evaluate(board) is not None:
                print("Game over!")
                break
        else:
            print("Invalid move. Please try again.")

play_game()
