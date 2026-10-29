# Multi-Domain Image–Caption Inconsistency Detection

Hamza Javid · Student ID 26009900 · MSc Artificial Intelligence and Data Science · CSC-44120 · Keele University

A local web application that checks whether a short claim ("The car is an SUV.", "The dog is a beagle.",
"The flower is a sunflower.", "The shape is on the left.") agrees with an uploaded photograph. The model is a
dual encoder trained on five domains (cars, dogs, cats, flowers, generated shapes) with no photographs of people.
No language model is used anywhere in the system.

## Run the website

Requires Python 3.10+. Runs on CPU; no GPU needed.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 BACKEND/app.py --model results_pretrained_224/model.pt
```

Open http://127.0.0.1:8000. Choose a domain, upload a photo (samples in `demo_photos/`), pick or type a claim,
and press Check. If port 8000 is busy, add `--port 8001`.

## Project structure

| Path | Contents |
| --- | --- |
| `BACKEND/app.py` | FastAPI server: model loading, sentence matching, auto-crop, domain check |
| `frontend/index.html` | The web page |
| `training/model.py`, `training/train.py` | Model and training/evaluation script |
| `process_all_datasets.py`, `person_filter.py` | Data pipeline and person screening |
| `evaluation/` | CLIP baseline and metric comparison (accuracy, precision, recall, F1, ROC-AUC) |
| `Data/generate_objects_dataset.py` | Generator for the shapes domain |
| `results_pretrained_224/` | Final model, metrics and test predictions |
| `processed_data/` | Train/val/test claim files used for the reported results |

## Results (test set, chance = 50%)

| Model | Overall | Cars | Dogs | Cats | Flowers | Shapes |
| --- | --- | --- | --- | --- | --- | --- |
| From scratch (run 7) | 78.7% | 70.6% | 61.5% | — | 86.9% | 99.9% |
| Pretrained ResNet-18, 224 px (final) | **95.3%** | 89.5% | 95.7% | 97.7% | 94.1% | 100% |

## Retraining (optional)

The raw datasets are not included (size and licences). Download Stanford Cars, Stanford Dogs,
Microsoft Cats and Dogs (Cat folder) and Oxford 102 Flowers into `Data/`, then:

```bash
python3 Data/generate_objects_dataset.py
python3 process_all_datasets.py --skip-coco
python3 training/train.py --backbone resnet18 --epochs 15 --image-size 224 --out results_pretrained_224
```

## Ethics

1,521 photographs containing people were removed with a person detector before training.
Uploaded photos are processed in memory and never saved; the server listens on 127.0.0.1 only.
