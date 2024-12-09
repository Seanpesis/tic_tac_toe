import cv2
import numpy as np
import mediapipe as mp
import time
import math
import random

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
STREAM_WIDTH = 1000
STREAM_HEIGHT = 800
SCREEN_CENTER_X = SCREEN_WIDTH // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2
CELL_SIZE = 200
BOARD_START_X = (STREAM_WIDTH - (CELL_SIZE * 3)) // 2
BOARD_START_Y = (STREAM_HEIGHT - (CELL_SIZE * 3)) // 2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

board = [["" for _ in range(3)] for _ in range(3)]
user_turn = True
game_over = False
winner = None
selected_cell = None
cell_selected_for_confirmation = False
last_point_time = 0
hold_time_for_confirmation = 1.0
state = "start"
user_mark = ""
computer_mark = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

cv2.namedWindow("Tic-Tac-Toe", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tic-Tac-Toe", STREAM_WIDTH, STREAM_HEIGHT)
cv2.moveWindow("Tic-Tac-Toe", SCREEN_CENTER_X - STREAM_WIDTH // 2, SCREEN_CENTER_Y - STREAM_HEIGHT // 2)

def check_winner(b):
    for i in range(3):
        if b[i][0] != "" and b[i][0] == b[i][1] == b[i][2]:
            return b[i][0]
    for j in range(3):
        if b[0][j] != "" and b[0][j] == b[1][j] == b[2][j]:
            return b[0][j]
    if b[0][0] != "" and b[0][0] == b[1][1] == b[2][2]:
        return b[0][0]
    if b[0][2] != "" and b[0][2] == b[1][1] == b[2][0]:
        return b[0][2]
    return None

def board_full(b):
    for i in range(3):
        for j in range(3):
            if b[i][j] == "":
                return False
    return True

def random_computer_move(b, m):
    empty = []
    for i in range(3):
        for j in range(3):
            if b[i][j] == "":
                empty.append((i, j))
    if not empty:
        return
    c = random.choice(empty)
    b[c[0]][c[1]] = m

def reset_game():
    for i in range(3):
        for j in range(3):
            board[i][j] = ""
    return True, False, None, False, None, False, 0, "", ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if state == "start":
        display_frame = np.ones((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8) * 255
        start_text = "Press 'S' to start"
        (tw, th) = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        tx = (STREAM_WIDTH - tw) // 2
        ty = STREAM_HEIGHT // 2
        cv2.putText(display_frame, start_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        made_by = "Created by Sean Pesis"
        (mw, mh) = cv2.getTextSize(made_by, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        mx = (STREAM_WIDTH - mw) // 2
        my = STREAM_HEIGHT - 50
        cv2.putText(display_frame, made_by, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Tic-Tac-Toe", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            user_turn, game_over, winner, cell_selected_for_confirmation, selected_cell, cell_selected_for_confirmation, last_point_time, user_mark, computer_mark = reset_game()
            state = "choose_symbol"

    elif state == "choose_symbol":
        display_frame = np.ones((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8) * 255
        choose_text = "Choose symbol: Press 'O' for O or 'X' for X"
        font_scale = 1.0
        (tw, th) = cv2.getTextSize(choose_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        tx = (STREAM_WIDTH - tw) // 2
        ty = STREAM_HEIGHT // 2
        cv2.putText(display_frame, choose_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)

        cv2.imshow("Tic-Tac-Toe", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('o'):
            user_mark = "O"
            computer_mark = "X"
            state = "game"
        elif k == ord('x'):
            user_mark = "X"
            computer_mark = "O"
            state = "game"
        elif k == ord('q'):
            break

    elif state == "game":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        display_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))

        instruction_text = f"Point for 1 second inside a cell to place {user_mark}"
        (iw, ih) = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        ix = (STREAM_WIDTH - iw) // 2
        iy = BOARD_START_Y - 60
        cv2.putText(display_frame, instruction_text, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for i in range(4):
            cv2.line(display_frame, (BOARD_START_X + i * CELL_SIZE, BOARD_START_Y),
                     (BOARD_START_X + i * CELL_SIZE, BOARD_START_Y + 3 * CELL_SIZE), (255, 255, 255), 5)
            cv2.line(display_frame, (BOARD_START_X, BOARD_START_Y + i * CELL_SIZE),
                     (BOARD_START_X + 3 * CELL_SIZE, BOARD_START_Y + i * CELL_SIZE), (255, 255, 255), 5)

        for i in range(3):
            for j in range(3):
                cx = BOARD_START_X + j * CELL_SIZE + CELL_SIZE // 2
                cy = BOARD_START_Y + i * CELL_SIZE + CELL_SIZE // 2
                if board[i][j] == "X":
                    cv2.putText(display_frame, "X", (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                elif board[i][j] == "O":
                    cv2.putText(display_frame, "O", (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        if not game_over and user_turn:
            hand_info = None
            if res.multi_hand_landmarks:
                hlms = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display_frame, hlms, mp_hands.HAND_CONNECTIONS)
                x_index = int(hlms.landmark[8].x * STREAM_WIDTH)
                y_index = int(hlms.landmark[8].y * STREAM_HEIGHT)
                cv2.circle(display_frame, (x_index, y_index), 10, (0, 255, 0), -1)
                hand_info = (x_index, y_index)

            if not cell_selected_for_confirmation:
                if hand_info:
                    x_i, y_i = hand_info
                    col = (x_i - BOARD_START_X) // CELL_SIZE
                    row = (y_i - BOARD_START_Y) // CELL_SIZE
                    if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == "":
                        cell_x1 = BOARD_START_X + col * CELL_SIZE
                        cell_y1 = BOARD_START_Y + row * CELL_SIZE
                        cell_x2 = cell_x1 + CELL_SIZE
                        cell_y2 = cell_y1 + CELL_SIZE
                        cv2.rectangle(display_frame, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 255, 0), 3)
                        if selected_cell != (row, col):
                            selected_cell = (row, col)
                            last_point_time = time.time()
                            cv2.putText(display_frame, "Hold finger still...", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        else:
                            elapsed = time.time() - last_point_time
                            if elapsed < hold_time_for_confirmation:
                                cv2.putText(display_frame, "Holding...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), 2)
                            elif elapsed > hold_time_for_confirmation:
                                r, c = selected_cell
                                if board[r][c] == "":
                                    board[r][c] = user_mark
                                    selected_cell = None
                                    cell_selected_for_confirmation = False
                                    user_turn = False
                                    w = check_winner(board)
                                    if w is not None:
                                        winner = w
                                        game_over = True
                                    elif board_full(board):
                                        game_over = True
                                    else:
                                        random_computer_move(board, computer_mark)
                                        w = check_winner(board)
                                        if w is not None:
                                            winner = w
                                            game_over = True
                                        elif board_full(board):
                                            game_over = True
                                        else:
                                            user_turn = True
                    else:
                        selected_cell = None
                        last_point_time = 0
                else:
                    selected_cell = None
                    last_point_time = 0

        if game_over:
            state = "result"

        cv2.imshow("Tic-Tac-Toe", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    elif state == "result":
        result_frame = np.ones((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8) * 255
        if winner is not None:
            if winner == user_mark:
                msg = "You won!"
            else:
                msg = "Computer won!"
        else:
            msg = "Draw!"
        (tw, th) = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        tx = (STREAM_WIDTH - tw) // 2
        ty = STREAM_HEIGHT // 2
        cv2.putText(result_frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        info = "Press 'q' to quit or 's' for a new game"
        (iw, ih) = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        ix = (STREAM_WIDTH - iw) // 2
        iy = ty + 100
        cv2.putText(result_frame, info, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Tic-Tac-Toe", result_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            user_turn, game_over, winner, cell_selected_for_confirmation, selected_cell, cell_selected_for_confirmation, last_point_time, user_mark, computer_mark = reset_game()
            state = "choose_symbol"

cap.release()
cv2.destroyAllWindows()
