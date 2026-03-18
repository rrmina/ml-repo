# Project Framework

This document provides a pedagogical guide to understanding the structure and workflow of a machine learning project.

## Overview

This project follows a modular design pattern that separates concerns into distinct components. Each file has a specific responsibility, making the codebase easier to understand, test, and maintain. Think of it as an assembly line: data flows through different stages, each handled by a specialized module.

## Project Structure

```
my_project/
│
├── data.py           # Step 1: Get and prepare your data
├── models.py         # Step 2: Define your model architecture
├── train.py          # Step 3: Train your model
├── eval.py           # Step 4: Evaluate performance
├── main.py           # Orchestrator: Brings everything together
├── inference.py      # Demo: See your model in action
│
├── requirements.txt  # Dependencies needed
├── checkpoints/      # Saved model weights
└── data/            # Raw and processed datasets
```

## Workflow: How Everything Connects

```
data.py → models.py → train.py → eval.py → inference.py
   ↓          ↓           ↓          ↓           ↓
[Load]   [Define]   [Optimize]  [Measure]   [Predict]
```

All coordinated through **main.py**

---

## Component Details

### 📦 `requirements.txt`
**Purpose:** Dependency management

Contains all Python packages your project needs (e.g., `torch`, `numpy`, `pandas`). This ensures anyone can recreate your environment with `pip install -r requirements.txt`.

**Why it matters:** Reproducibility is crucial in ML projects. This file documents exactly which versions of libraries were used.

---

### 📊 `data.py`
**Purpose:** Data loading and preprocessing

This is where raw data becomes ML-ready data. Think of it as the kitchen prep before cooking.

**Key responsibilities:**
- **Loading:** Read data from files (CSV, JSON, databases)
- **Preprocessing:** Clean missing values, normalize features, encode categories
- **Validation:** Check data quality and integrity
- **Utilities:** Create data loaders, batching, train/test splits

**Example functions:**
```python
# Reads raw data
def load_dataset(
    data_path: str
) -> Any:
    
    # TODO: Implement data loading
    pass

# Cleans and transforms
def preprocess_data(
    raw_data: Any
) -> Any:
    
    # TODO: Implement preprocessing
    pass

# Prepares batches for training
def create_dataloaders(
    processed_data: Any,
    batch_size: int = 32
) -> Tuple[Any, Any]:
    
    # TODO: Implement dataloader creation
    pass
```

**Connection:** Feeds processed data to `train.py` and `eval.py`

---

### 🏗️ `models.py`
**Purpose:** Model architecture definitions

This module contains the blueprint of your ML models. It's like an architect's drawing before construction begins.

**Key responsibilities:**
- **Architecture:** Define neural network layers, structure
- **Classes:** Implement model classes (e.g., `MyNeuralNetwork`, `TransformerModel`)
- **Configuration:** Set hyperparameters (layers, dimensions, activation functions)

**Example structure:**
```python
# Neural network model class
class MyModel(nn.Module):
    # Initialize model layers
    def __init__(self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10
    ) -> None:
        
        # TODO: Define layers
        pass
    
    # Define forward pass
    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        # TODO: Define forward pass logic
        pass
```

**Connection:** Used by `train.py` to instantiate and train, by `eval.py` to test, and by `inference.py` to make predictions

---

### 🎯 `train.py`
**Purpose:** Training logic

This is where the learning happens. The model adjusts its weights to minimize errors on your data.

**Key responsibilities:**
- **Training loop:** Iterate over data, compute loss, update weights
- **Optimization:** Choose optimizer (Adam, SGD), learning rate
- **Checkpointing:** Save model weights periodically
- **Logging:** Track loss, accuracy over epochs

**Typical flow:**
1. Initialize model from `models.py`
2. Load data from `data.py`
3. For each epoch:
   - Forward pass → compute predictions
   - Calculate loss
   - Backward pass → compute gradients
   - Update weights
4. Save trained model

**Connection:** Takes models and data, produces trained weights saved to `checkpoints/`

---

### 📈 `eval.py`
**Purpose:** Performance evaluation

After training, you need to know: "How good is my model?" This module answers that question.

**Key responsibilities:**
- **Metrics:** Calculate accuracy, precision, recall, F1, etc.
- **Testing:** Run model on held-out test data
- **Reporting:** Generate performance summaries
- **Analysis:** Confusion matrices, error analysis

**Why separate from training?**
- Keeps code clean and focused
- Allows evaluation without retraining
- Makes it easy to compare different models

**Connection:** Loads trained models from `checkpoints/`, uses test data from `data.py`

---

### 🎬 `main.py`
**Purpose:** Central orchestration

This is mission control. Almost all modules are imported here, and the entire pipeline is coordinated from this script.

**Key responsibilities:**
- **Entry point:** The script you run to start everything
- **Integration:** Imports and connects all components
- **Workflow:** `data.py` → `models.py` → `train.py` → `eval.py`
- **CLI:** Parse command-line arguments (e.g., `--mode train`, `--epochs 50`)

**Typical usage:**
```bash
python main.py --mode train   # Train a model
python main.py --mode eval    # Evaluate a model
```

**Why it's important:** Provides a single, clear interface to your entire project. Instead of running scripts individually, you control everything from here.

**Note:** The generated `main.py` includes commented example code in each TODO section showing how to use all the imported functions (`load_dataset`, `preprocess_data`, `create_dataloaders`, `MyModel`, `train_model`, `evaluate_model`, `run_inference`). These examples serve as a guide for implementing each workflow mode.

**Example implementation:**
```python
# main.py - Import all modules at the top
import argparse
import torch

# Import all project modules
from data import load_dataset, preprocess_data, create_dataloaders
from models import MyModel
from train import train_model
from eval import evaluate_model
from inference import run_inference


def main(args):
    """Main function that orchestrates the entire workflow."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Step 1: Load and prepare data
        print("Loading data...")
        raw_data = load_dataset(args.data_path)
        processed_data = preprocess_data(raw_data)
        train_loader, val_loader = create_dataloaders(
            processed_data, 
            batch_size=args.batch_size
        )
        
        # Step 2: Initialize model
        print("Initializing model...")
        model = MyModel(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        ).to(device)
        
        # Step 3: Train model
        print("Starting training...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            save_path=args.checkpoint_path
        )
        
    elif args.mode == 'eval':
        # Load trained model and evaluate
        print("Loading model for evaluation...")
        model = MyModel().to(device)
        model.load_state_dict(torch.load(args.checkpoint_path))
        
        # Load test data
        raw_data = load_dataset(args.data_path)
        processed_data = preprocess_data(raw_data)
        _, test_loader = create_dataloaders(processed_data)
        
        # Evaluate
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device)
        print(f"Evaluation Results: {results}")
        
    elif args.mode == 'inference':
        # Run inference demo
        print("Running inference...")
        run_inference(
            checkpoint_path=args.checkpoint_path,
            input_data=args.input,
            device=device
        )
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ML Project Main Script')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'eval', 'inference'],
                       help='Mode: train, eval, or inference')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/dataset.csv',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, 
                       default='./checkpoints/model.pth',
                       help='Path to save/load model checkpoint')
    
    # Inference arguments
    parser.add_argument('--input', type=str, default=None,
                       help='Input for inference mode')
    
    args = parser.parse_args()
    main(args)
```

---

### 🔮 `inference.py`
**Purpose:** Demonstration and prediction

This is your model's "showtime" - where you demonstrate what it can do on new, unseen data.

**Key responsibilities:**
- **Demo:** Show model capabilities with examples
- **Prediction:** Generate outputs for new inputs
- **Testing:** Quick sanity checks on trained models
- **Deployment preview:** Simulate how the model would work in production

**Example use cases:**
- Generate text with a language model
- Classify new images
- Make predictions on custom inputs
- Create visualizations of model outputs

**Connection:** Loads trained model from `checkpoints/`, may use utilities from `data.py` for preprocessing

---

## 🚀 Getting Started

### Step-by-step workflow:

1. **Setup environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**
   - Place raw data in `data/` folder
   - Update `data.py` with your data loading logic

3. **Define your model**
   - Create model class in `models.py`

4. **Train your model**
   ```bash
   python main.py --mode train
   ```

5. **Evaluate performance**
   ```bash
   python main.py --mode eval
   ```

6. **Try inference**
   ```bash
   python inference.py
   ```

---

## 💡 Key Principles

**Modularity:** Each file has one clear purpose
**Separation of Concerns:** Data, models, training, and evaluation are independent
**Reusability:** Components can be mixed and matched for different projects
**Maintainability:** Easy to debug, test, and extend
**Reproducibility:** Clear structure makes results reproducible

---

## 📁 Directory Reference

- **`checkpoints/`**: Stores saved model weights during training
- **`data/`**: Contains raw and processed datasets
