from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Corrected the value from "True" to ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow SavedModel as an inference-only layer
inference_layer = tf.keras.layers.TFSMLayer("../saved-models/1", call_endpoint="serving_default")

# Create a Keras model with the inference-only layer
inputs = tf.keras.Input(shape=(256, 256, 3))  # Define your input shape
outputs = inference_layer(inputs)
model = tf.keras.Model(inputs, outputs)

# Compile the model (necessary even for inference)
model.compile()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    # Open image and resize to 256x256, convert to RGB
    image = Image.open(BytesIO(data)).convert('RGB').resize((256, 256))
    image = np.array(image)
    print(f"Image shape: {image.shape}")  # Debugging shape
    return image

@app.post("/")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Debugging batch shape
    print(f"Batch shape: {img_batch.shape}")

    # Make sure to normalize the input image data if needed
    # For example: img_batch = img_batch / 255.0

    try:
        predictions = model.predict(img_batch)
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": "Prediction error occurred"}

    # Debug predictions
    print(f"Predictions: {predictions}")

    predicted_class_index = np.argmax(predictions['dense_1'][0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions['dense_1'][0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
