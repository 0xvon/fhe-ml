from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mnist import TinyCNN
import numpy as np
from PIL import Image
import io
import base64
from concrete.ml.torch.compile import compile_torch_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

X, y = load_digits(return_X_y=True)
X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42
)

model = TinyCNN(10)
model.load_state_dict(torch.load("mnist_state_dict.pt"))

q_model = compile_torch_model(model, x_train, rounding_threshold_bits=6, p_error=0.1)

@app.get("/img/")
async def get_img(id: int):
    X, y = img(id)
    X_img = img_to_base64(X[0])
    
    return { "id": id, "X": X[0].tolist(), "X_img": X_img,  "y": y.tolist() }

@app.post("/predict/")
async def predict(id: int):
    X, y = img(id)
    # input_tensor = torch.tensor(X, dtype=torch.float32).view(1,1,8,8)

    q_model.fhe_circuit.keygen()
    output = q_model.forward(X.reshape(1, 1, 8, 8), fhe="execute").argmax(1)
    # output = model(input_tensor).argmax(1).detach().numpy()

    return { "output": output.tolist()[0], "ans": y.tolist() }

@app.post("/predict_raw/")
async def predict(id: int):
    X, y = img(id)
    input_tensor = torch.tensor(X, dtype=torch.float32).view(1,1,8,8)
    output = model(input_tensor).argmax(1).detach().numpy()

    return { "output": output.tolist()[0], "ans": y.tolist() }

def img(id: int):
    return (X[id], y[id])

def img_to_base64(img_array):
    img_array = np.round(img_array * (255 / 16)).astype(np.uint8)
    pil_image = Image.fromarray(img_array)

    # バイナリストリームに保存
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    # Base64文字列にエンコード
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return encoded_image