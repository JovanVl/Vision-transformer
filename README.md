# Vision-transformer
An implementation of the Vision transformer architecture and its use for artwork period image classification

## Course:
Computer intelligence

## Author:
Jovan Vlatković 176/2021,

Faculty of Mathematics

## Project structure
 
```
project/
├── data/            # directory for raw, processed and train/test data
├── docs/            # report & presentation
├── notebooks/       # training and evaluating the model
├── src/
│   ├── models/
│   │   ├── ViT/
│   │   │   └── vit.py          # ViT, Encoder, EncoderLayer, MultiHeadAttention, PatchEmbedding
│   │   └── CNN/
│   │       └── simple_cnn.py   # SimpleCNN baseline
│   ├── data/
│   │   ├── preprocessing.py                  # resize + balance raw images
│   │   ├── image_dataset_train_test_split.py # stratified train/test split
│   │   └── image_dataset.py                  # ImageDataset (PyTorch Dataset)
│   └── training/
│       └── loop.py             # train_epoch, evaluate, fit, predict
└── pyproject.toml
```
 
## Installation
 
```bash
git clone https://github.com/JovanVl/Vision-transformer.git
cd Vision-transformer
pip install -e .
pip install torch torchvision pillow numpy matplotlib scikit-learn pandas
```
