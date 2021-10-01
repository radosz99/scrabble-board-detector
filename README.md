# General info
RPC API created using FastAPI for detecting letters based on image with scrabble board in it.

# Endpoints
<a name="best"></a>
## Get letters pattern (POST)
The only endpoint in API requires an image in request body.
### URL
```
http://127.0.0.1:8000/detect_letters
```
### Example using Python requests
In request there must be parameter `files` included, with key `image` and binary representation of image as value.
```
import requests
import ast

IMG_PATH = "test.jpg"
SERVER_URL = "http://127.0.0.1:8000"

files = {'image': open(IMG_PATH,'rb')}
response = requests.post(url=f"{SERVER_URL}/detect_letters", files=files)
json_response = ast.literal_eval(response.text)
board = json_response['board']
```
As a response there is a json with only one key-value pair - `board` and its representation as a list of list:
```
{
  "board": [[' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', 'o', 'p', 'e', 'r', 'a', 't',  'i', 'o', 'n', 'a', 'l', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]
}
```

# Run
From the root folder:
## FastAPI (Windows)
```
uvicorn project.main:app --reload
```
# Algorithm
