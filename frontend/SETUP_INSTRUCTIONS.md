# Multi-Domain Inconsistency Detector - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- Your datasets downloaded in `~/my-dissertation-v2/data/`

---

## 📁 File Placement

Copy files to your project:

```
~/my-dissertation-v2/
├── backend/
│   ├── app.py              ← Paste: backend_app.py
│   └── requirements.txt     ← Paste: requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx         ← Paste: frontend_App.jsx (replace existing)
│   │   ├── main.jsx        (keep existing)
│   │   └── style.css       (keep existing)
│   ├── index.html
│   └── package.json        (keep existing)
├── processed_data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── data/
    ├── stanford_cars/
    ├── oxford_flowers/
    ├── cats_dogs/
    └── synthetic_objects/
```

---

## 🔧 Backend Setup

### 1. Create Python environment
```bash
cd ~/my-dissertation-v2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Run backend server
```bash
python app.py
```

**Expected output:**
```
✓ Model loaded successfully
Uvicorn running on http://127.0.0.1:8000
```

---

## 🎨 Frontend Setup

### 1. Install Node dependencies
```bash
cd ~/my-dissertation-v2/frontend
npm install
```

### 2. Build frontend
```bash
npm run build
```

### 3. Serve frontend
```bash
npm run dev
```

**Expected output:**
```
Local: http://localhost:5173
```

---

## ✅ Testing

Once both are running, open browser:

```
http://localhost:5173
```

You should see:
- ✓ Status indicator shows "Ready"
- Domain selection buttons (Cars, Dogs, Flowers, Objects)
- Image upload area
- Caption input field
- Analyze button

---

## 🧠 Using Your Trained Model

When you train your custom model, save checkpoint:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'vocab_size': vocab_size,
}, '~/my-dissertation-v2/model.pth')
```

Backend will automatically load it!

---

## 📊 Sample Workflow

1. **Run backend**: `python ~/my-dissertation-v2/backend/app.py`
2. **Run frontend**: `npm run dev` in `frontend/`
3. **Test predictions**:
   - Select domain (e.g., "Cars")
   - Upload image from that domain
   - Write caption: "The car is red"
   - Click "Analyze Image & Caption"
   - Get result: "Consistent" or "Inconsistent"

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Check port 8000 is free
lsof -i :8000  # Kill with: kill -9 <PID>
```

### Frontend won't connect to backend
```bash
# Check backend is running on port 8000
curl http://127.0.0.1:8000/api/health

# Check CORS is enabled (it is by default)
```

### Model not loading
```bash
# Demo model should work
# If using custom model, ensure checkpoint exists at:
~/my-dissertation-v2/model.pth
```

---

## 📝 Next Steps

1. ✅ Process datasets with `process_all_datasets.py`
2. ✅ Set up frontend & backend (this guide)
3. ⏭️ Train your custom model on processed data
4. 🔄 Replace demo model with trained checkpoint
5. 📊 Evaluate on test set

---

## 🎯 API Endpoints

- `GET /api/health` - Check model status
- `POST /api/predict` - Get prediction for image+caption
- `GET /api/samples` - Get test samples
- `GET /api/image/{path}` - Serve images

---

**Questions? Check your dissertation project structure matches the layout above!**
