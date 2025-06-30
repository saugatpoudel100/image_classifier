from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import joblib
import io
import os

app = FastAPI()

# Serve static HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load ML model
model = joblib.load("app/model/animal_model.pkl")
encoder = joblib.load("app/model/label_encoder.pkl")

IMG_SIZE = (64, 64)

@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("static/upload.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = imread(io.BytesIO(contents))
    img_resized = resize(img, IMG_SIZE).flatten().reshape(1, -1)
    prediction = model.predict(img_resized)
    label = encoder.inverse_transform(prediction)[0]
    return JSONResponse(content={"prediction": label})
