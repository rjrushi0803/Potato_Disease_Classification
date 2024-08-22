from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

model_path = r"D:/VS_code/R/New folder/potato disease_pred/saved_model_1/1"
MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']

@app.get("/ping")
async def ping():
    return 'Hello Server is Alive'

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    File:UploadFile = File(...)
    ):
    image = await File.read()

    img_batch = np.expand_dims(image,0)
    prediction=MODEL.predict(img_batch)
    pass

if __name__=='__main__':
    uvicorn.run(app, host='localhost',port=8000)