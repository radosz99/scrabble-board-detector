import requests
import ast
import copy
from project.main import app
from fastapi.testclient import TestClient

DETECTOR_SERVER_URL = 'http://127.0.0.1:8000'
CHEATER_SERVER_URL = 'http://127.0.0.1:7088'

letters = "ABTUKLO"
ASCII_CONST = 65

client = TestClient(app)

def test_e2e():
    files = {'image': open('resources/test.jpg','rb')}
    response = client.post(f"/detect_letters", files=files)
    assert response.status_code == 200
    response = ast.literal_eval(response.text)
    assert 'board' in response
    board = response['board']
    assert board
    print(board)
    json_body = {"letters": letters, "board": board}
    response = requests.post(url=f"{CHEATER_SERVER_URL}/best-move/GB", json=json_body)
    assert response.status_code == 200
    response = ast.literal_eval(response.text)
    assert 'moves' in response
    moves = response['moves']
    updated_board = make_move(moves[0], board)
    assert board != updated_board

def make_move(best_move, board):
    _board = copy.deepcopy(board)
    coordinates, word, points = best_move['coordinates'], best_move['word'], best_move['points']
    coord_x, coord_y, orientation = get_move_details_based_on_scrabble_coordinates(coordinates)
    print(f"x - {coord_x}, y - {coord_y}, orient - {orientation}, word - {word}, points - {points}")
    put_word_on_board(word, coord_x, coord_y, orientation, _board)
    return _board

def get_move_details_based_on_scrabble_coordinates(coordinates):
    coords = coordinates.split('_')  # 10_F -> [10,F]
    coord_x, coord_y = 0, 0
    orientation = 1  # 0 - vertical, 1 - horizontal
    try:
        coord_x = int(coords[0])
        coord_y = ord(coords[1][0]) - ASCII_CONST
    except Exception:
        coord_y = ord(coords[0][0]) - ASCII_CONST
        coord_x = int(coords[1])
        orientation = 0
    return coord_x, coord_y, orientation

def put_word_on_board(word, coord_x, coord_y, orientation, board):
    if(orientation==1):
        for i, char in enumerate(word):
            board[coord_x][coord_y + i] = char
    elif(orientation==0):
        for i, char in enumerate(word):
            board[coord_x + i][coord_y] = char

def retrieve_board_from_image(img_path):
    files = {'image': open(img_path,'rb')}
    response = requests.post(url=f"{DETECTOR_SERVER_URL}/detect_letters", files=files)
    json_response = ast.literal_eval(response.text)
    return json_response['board']

def get_best_moves(json_body):
    response = requests.post(url=f"{CHEATER_SERVER_URL}/best-move/GB", json=json_body)
    json_response = ast.literal_eval(response.text)
    return json_response['moves']

def basic_e2e_test():
    board = retrieve_board_from_image('resources/test.jpg')
    json_body = {"letters": letters, "board": board}
    best_move = get_best_moves(json_body)[0]
    updated_board = make_move(best_move, board)

def show_board(board):
    for row in board:
        print(row)
