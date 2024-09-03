from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer,ViTImageProcessor
from PIL import Image
import io

app = FastAPI()

# Load model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("../VIT_large_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    

    generated_caption = tokenizer.decode(model.generate(feature_extractor(image, return_tensors="pt").pixel_values.to("cpu"))[0])
    # Decode output

    return {"caption":generated_caption}
