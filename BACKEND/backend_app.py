"""
FastAPI Backend for Multi-Domain Image-Caption Inconsistency Detection
Supports: Cars, Dogs, Flowers, Objects
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import torch
import json
from pathlib import Path
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = Path("model.pth")  # We'll use this when you train
PROCESSED_DATA_DIR = Path("processed_data")
PROJECT_DIR = Path.home() / "my-dissertation-v2"

# Domain configuration
DOMAINS = {
    'cars': {'colors': ['red', 'blue', 'silver', 'black', 'white', 'grey']},
    'dogs': {'colors': ['brown', 'black', 'white', 'golden', 'grey']},
    'flowers': {'colors': ['red', 'yellow', 'pink', 'white', 'purple', 'orange']},
    'objects': {'colors': ['red', 'blue', 'green', 'yellow', 'orange', 'purple']},
}

# Global model state
model_state = {
    'loaded': False,
    'model': None,
    'vocab': None,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

class SimpleModel(torch.nn.Module):
    """Simplified model for demo - replace with your trained model"""
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.text_embedding = torch.nn.Embedding(vocab_size, 128)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(128 + 128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, image, text_tokens):
        img_feat = self.image_encoder(image).flatten(1)  # (B, 128)
        text_feat = self.text_embedding(text_tokens).mean(dim=1)  # (B, 128)
        combined = torch.cat([img_feat, text_feat], dim=1)  # (B, 256)
        output = self.fusion(combined)  # (B, 1)
        return output

def load_model():
    """Load or initialize model"""
    global model_state

    try:
        if MODEL_PATH.exists():
            print(f"Loading model from {MODEL_PATH}...")
            checkpoint = torch.load(MODEL_PATH, map_location=model_state['device'])
            model = SimpleModel(vocab_size=checkpoint.get('vocab_size', 1000))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model_state['model'] = model
            model_state['vocab'] = checkpoint.get('vocab', {})
            model_state['loaded'] = True
            print("✓ Model loaded successfully")
            return True
        else:
            print(f"⚠ Model not found at {MODEL_PATH}")
            print("  Creating demo model for testing...")
            model = SimpleModel()
            model.eval()
            model_state['model'] = model
            model_state['loaded'] = True
            return True
    except Exception as e:
        print(f"Error loading model: {e}")
        model_state['loaded'] = False
        return False

def tokenize(text, vocab):
    """Simple tokenizer"""
    if not vocab:
        return [1] * len(text.split())  # Fallback
    tokens = []
    for word in text.lower().split():
        tokens.append(vocab.get(word, vocab.get('<unk>', 0)))
    return tokens[:24] + [vocab.get('<pad>', 0)] * (24 - len(tokens))  # Max 24 tokens

def preprocess_image(image: Image.Image):
    """Preprocess image to 256x256"""
    image = image.convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor.to(model_state['device'])

@app.on_event("startup")
async def startup():
    """Initialize model on startup"""
    load_model()

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_ready": model_state['loaded'],
        "device": str(model_state['device']),
        "domains": list(DOMAINS.keys()),
    }

@app.post("/api/predict")
async def predict(
    domain: str = Form(...),
    image: UploadFile = File(...),
    caption: str = Form(...),
):
    """
    Predict if caption is consistent with image

    Returns:
    - prediction: 'consistent' or 'inconsistent'
    - confidence: float 0-1
    - score: raw model score
    """
    try:
        # Validate domain
        if domain not in DOMAINS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid domain. Choose from: {list(DOMAINS.keys())}"}
            )

        # Validate caption length
        if len(caption) > 180:
            return JSONResponse(
                status_code=400,
                content={"error": "Caption must be ≤180 characters"}
            )

        if not model_state['loaded']:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not ready"}
            )

        # Read and preprocess image
        image_data = await image.read()
        image_obj = Image.open(io.BytesIO(image_data))
        img_tensor = preprocess_image(image_obj)

        # Tokenize caption
        tokens = tokenize(caption, model_state['vocab'])
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=model_state['device'])

        # Get prediction
        with torch.no_grad():
            raw_score = model_state['model'](img_tensor, tokens_tensor).item()

        # Threshold at 0.5
        threshold = 0.5
        prediction = 'consistent' if raw_score > threshold else 'inconsistent'

        return {
            "prediction": prediction,
            "confidence": round(abs(raw_score - 0.5) * 2, 4),
            "score": round(raw_score, 4),
            "domain": domain,
            "caption": caption,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/samples")
async def samples():
    """Return sample images and captions for testing"""
    samples_list = []

    # Try to load from processed_data
    try:
        test_csv = PROJECT_DIR / "processed_data" / "test.csv"
        if test_csv.exists():
            import pandas as pd
            df = pd.read_csv(test_csv).head(10)
            for idx, row in df.iterrows():
                samples_list.append({
                    "id": idx,
                    "domain": row['dataset'],
                    "image_url": f"/api/image/{row['image_path']}",
                    "caption": row['caption'],
                    "label": 'consistent' if row['label'] == 1 else 'inconsistent',
                })
    except Exception as e:
        print(f"Could not load samples: {e}")

    return {"samples": samples_list}

@app.get("/api/image/{file_path:path}")
async def get_image(file_path: str):
    """Serve image from dataset"""
    try:
        full_path = PROJECT_DIR / "data" / file_path
        if not full_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "Image not found"}
            )
        return JSONResponse(
            status_code=200,
            content={"path": str(full_path)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
