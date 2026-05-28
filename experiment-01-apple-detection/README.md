# Apple Quality Detection - Machine Learning Experiment

8-class image classification for apple quality/defect detection using PyTorch and ResNet18.

## Dataset

| Class (English) | Class (Chinese) | Samples |
|-----------------|-----------------|---------|
| fresh           | 合格             | 20      |
| diseased        | 病变             | 14      |
| bruised         | 碰伤             | 20      |
| rotten          | 腐烂             | 20      |
| insect_damaged  | 虫伤             | 20      |
| cracked         | 裂果             | 16      |
| wrinkled        | 褶皱             | 20      |
| black_spot      | 黑斑             | 20      |

**Total training samples:** ~170 images  
**Test samples:** 30 images (unlabeled, for prediction)

## Project Structure

```
ml-apple-detection/
├── data/
│   ├── train/              # Training data (organized by class)
│   ├── test/               # Test images to predict
│   └── raw/                # Original extracted data
├── models/                 # Saved model checkpoints
├── notebooks/
│   └── exploration.ipynb   # Data exploration notebook
├── src/
│   ├── train.py            # Training script (with text log export)
│   ├── predict.py          # Inference script
│   └── evaluate.py         # Validation evaluation & confusion matrix
├── outputs/                # Training logs, metrics JSON, and text logs
├── reports/                # Experiment report and generated figures
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Activate conda
source /opt/miniconda/bin/activate

# Install dependencies
conda install -y pytorch torchvision cpuonly -c pytorch
pip install opencv-python matplotlib scikit-learn tqdm jupyterlab tensorboard
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Exploration

```bash
cd notebooks
jupyter lab exploration.ipynb
```

### 2. Training

```bash
cd src
python train.py --epochs 50 --batch-size 8 --lr 0.001
```

**Training options:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.001)
- `--val-split`: Validation split ratio (default: 0.2)

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir=../outputs/runs
```

### 3. Evaluation (Confusion Matrix & Per-Class Metrics)

```bash
cd src
python evaluate.py --model ../models/best_model.pth --val-split 0.2
```

Outputs:
- Console: confusion matrix, classification report, overall accuracy
- `reports/figures/confusion_matrix.png`

### 4. Prediction

**Single image:**
```bash
python predict.py ../data/test/image.jpg --model ../models/best_model.pth
```

**Batch prediction (all test images):**
```bash
python predict.py ../data/test --model ../models/best_model.pth --output ../outputs/predictions.json
```

### 5. Generate Report Figures

After training, generate loss/accuracy/lr curves for the experiment report:

```bash
cd reports
python generate_plots.py
```

Outputs to `reports/figures/`:
- `loss_curve.png`
- `accuracy_curve.png`
- `lr_curve.png`

Also prints a markdown-ready summary table for pasting into the report.

### 6. Experiment Report

See `reports/experiment_report.md` for the report template. Fill in the results sections after training.

## Model Architecture

- **Backbone:** ResNet18 (pretrained on ImageNet)
- **Classifier:** Custom FC layer -> 8 classes
- **Input size:** 224x224
- **Data augmentation:** Random crop, horizontal flip, rotation, color jitter

## Expected Results

With the small dataset (~170 images), expected accuracy after 50 epochs:
- **Training accuracy:** ~90-95%
- **Validation accuracy:** ~75-85%

> Note: Results may vary due to the small dataset size. For production use, collect more training data.

## Tips for Better Performance

1. **Data augmentation:** Already implemented in train.py
2. **Transfer learning:** Using ImageNet pretrained weights
3. **Learning rate scheduling:** Cosine annealing
4. **Regularization:** Weight decay (L2) applied
5. **More data:** Collect additional images for underrepresented classes (e.g., diseased: 14 samples)

## Experiment Checklist

- [ ] Run data exploration notebook
- [ ] Train model and monitor with TensorBoard
- [ ] Evaluate on validation set
- [ ] Run predictions on test images
- [ ] Analyze misclassified examples
- [ ] Try different hyperparameters (lr, batch_size, epochs)
- [ ] (Optional) Try other architectures: ResNet34, EfficientNet

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [TensorBoard Tutorial](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
