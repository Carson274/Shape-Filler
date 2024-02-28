# Shape Filler
Generates random shapes including dots, lines, polygons, circles, and stars and trains a U-Net to fill them in.
Uses Pillow for image generation and PyTorch for designing the U-Net architecture.

## Try it Out
To get the code:
```
git pull https://github.com/Carson274/Shape-Filler.git
```
### From there, you can either:
1. Test the currently loaded model:
```
python3 test.py
```
2. Train a new model:
```
python3 train.py
```

## Tech Stack
- **PyTorch**: Neural network modeling and training.
- **Pillow (PIL)**: Image generation and processing.
- **NumPy**: Data manipulation and operations.
- **torchvision**: Image transformations and utilities.
- **matplotlib**: Plotting training loss graphs.
