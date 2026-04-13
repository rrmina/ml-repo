#!/bin/bash

# ML Research Project Setup Script
# 
# Creates a complete ML research project structure following the ML Research Framework.
# 
# Philosophy:
#   - What You See Is What You Get: Explicit and traceable code
#   - Prefer Explicit Pipelines over Magic: No hidden callbacks or hooks
#   - Keep Code Close to Research Ideas: Clear mapping from concepts to code
#   - Single Responsibility: Each module does one thing well
#
# This setup script uses UV for fast dependency management.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Get project name from user or use default
if [ -z "$1" ]; then
    read -p "Enter project name (default: my_ml_project): " PROJECT_NAME
    PROJECT_NAME=${PROJECT_NAME:-my_ml_project}
else
    PROJECT_NAME=$1
fi

print_status "Setting up ML project: $PROJECT_NAME"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    print_warning "UV is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | grep -v "Permission denied" || true
    
    # Try common UV installation paths
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Verify UV is now accessible
    if command -v uv &> /dev/null; then
        print_success "UV installed successfully ($(uv --version))"
        print_warning "Note: If UV is not found in future sessions, add this to your shell config:"
        echo "           export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        print_error "UV installation failed. Please install manually:"
        echo "           curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "           Then add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
else
    print_success "UV is already installed ($(uv --version))"
fi

# Initialize UV project
print_status "Initializing UV project..."
uv init "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Set Python version
print_status "Setting Python version to 3.12..."
echo "3.12" > .python-version

# Create project structure following ML Research Framework
print_status "Creating project structure..."

# Create core Python files (each with single responsibility)
touch train.py      # Orchestrates training, validation, and experiment flow
touch evaluate.py   # Handles evaluation of checkpoints independently
touch predict.py    # Handles inference on new inputs
touch losses.py     # Loss function definitions
touch metrics.py    # Reusable evaluation metrics
touch config.py     # Configuration schema and loading logic
touch utils.py      # Generic and reusable helper functions

# Create module directories for data and models
mkdir -p data       # Dataset, preprocessing, and dataloader logic
mkdir -p models     # Model architectures and forward computation

# Create __init__.py files for modules
touch data/__init__.py
touch models/__init__.py

# Create recommended directory structure
mkdir -p configs
mkdir -p tests
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p outputs/predictions
mkdir -p notebooks

# Remove default hello.py if it exists
if [ -f "hello.py" ]; then
    rm hello.py
fi

print_success "Project structure created (following ML Research Framework)"

# Create train.py with explicit, readable template
print_status "Creating train.py (main training orchestration)..."
cat > train.py << 'EOF'
# Training script - orchestrates training, validation, and experiment flow.
#
# This script should read like the main story of the experiment.
# All steps are explicit and traceable - no hidden magic or callbacks.

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from data import create_dataloaders
from model import MyModel
from losses import compute_loss
from metrics import compute_metrics
from utils import save_checkpoint, set_seed, setup_logging


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    # Train for one epoch - explicit and step-by-step.
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        loss = compute_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    # Validate model - explicit evaluation without hidden callbacks.
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    # Compute metrics explicitly
    val_loss = total_loss / len(val_loader)
    metrics = compute_metrics(
        torch.cat(all_outputs),
        torch.cat(all_targets)
    )
    metrics['val_loss'] = val_loss
    
    return metrics


def check_early_stopping(
    current_loss: float,
    best_loss: float,
    patience_counter: int,
    patience: int
) -> tuple[bool, int]:
    # Check early stopping condition explicitly.
    if current_loss < best_loss:
        return False, 0  # Continue training, reset patience
    else:
        patience_counter += 1
        if patience_counter >= patience:
            return True, patience_counter  # Stop training
        return False, patience_counter  # Continue training


def main(args: argparse.Namespace) -> None:
    # Main training function.
    #
    # Everything is visible and explicit:
    # - where data is loaded
    # - how the model is initialized
    # - how training is performed
    # - how validation happens
    # - when checkpoints are saved
    # Set seed for reproducibility
    set_seed(args.seed)
    setup_logging(args.log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data - explicit dataloader creation
    print('Loading data...')
    train_loader, val_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Initialize model - explicit architecture
    print('Initializing model...')
    model = MyModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Initialize optimizer - explicit configuration
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize learning rate scheduler (if needed)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop - explicit and easy to follow
    print(f'\\nStarting training for {args.epochs} epochs...\\n')
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device
        )
        val_loss = val_metrics['val_loss']
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f'\\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Metrics: {val_metrics}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            print(f'  → New best model! (prev: {best_val_loss:.4f})')
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                path=f'{args.checkpoint_dir}/best_model.pth'
            )
        
        # Check for early stopping - explicit condition
        should_stop, patience_counter = check_early_stopping(
            current_loss=val_loss,
            best_loss=best_val_loss,
            patience_counter=patience_counter,
            patience=args.patience
        )
        
        if should_stop:
            print(f'\\nEarly stopping triggered at epoch {epoch}')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
        
        # Save regular checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                path=f'{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
            )
    
    print('\\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ML model with explicit, traceable pipeline'
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./outputs/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_dir', type=str, 
                       default='./outputs/logs',
                       help='Directory for logs')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
EOF

print_success "train.py created"

# Create evaluate.py template
print_status "Creating evaluate.py (independent checkpoint evaluation)..."
cat > evaluate.py << 'EOF'
# Evaluation script - handles evaluation of checkpoints independently.
#
# Loads a trained checkpoint and evaluates it on test/validation data.
# All steps are explicit and easy to trace.

import argparse
import torch
import torch.nn as nn
from typing import Dict

from data import create_dataloaders
from model import MyModel
from metrics import compute_metrics
from utils import load_checkpoint


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    # Evaluate model on test set - explicit and traceable.
    model.eval()
    all_outputs = []
    all_targets = []
    
    print(f'Evaluating on {len(test_loader)} batches...')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            
            if batch_idx % 50 == 0:
                print(f'  Progress: {batch_idx}/{len(test_loader)}')
    
    # Compute all metrics
    metrics = compute_metrics(
        torch.cat(all_outputs),
        torch.cat(all_targets)
    )
    
    return metrics


def main(args: argparse.Namespace) -> None:
    # Main evaluation function.
    #
    # Load checkpoint, load data, run evaluation, report results.
    # Everything visible and explicit.
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading test data from {args.data_path}...')
    _, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f'Test batches: {len(test_loader)}')
    
    # Initialize model
    print('Initializing model...')
    model = MyModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint_path}...')
    checkpoint_data = load_checkpoint(args.checkpoint_path, device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint_data.get("epoch", "unknown")}')
    
    # Evaluate
    print('\\nRunning evaluation...')
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Print results
    print('\\n' + '='*50)
    print('Evaluation Results:')
    print('='*50)
    for metric_name, metric_value in metrics.items():
        print(f'  {metric_name}: {metric_value:.4f}')
    print('='*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained model checkpoint'
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    main(args)
EOF

print_success "evaluate.py created"

# Create predict.py template
print_status "Creating predict.py (inference on new inputs)..."
cat > predict.py << 'EOF'
# Prediction/Inference script - handles inference on new inputs.
#
# Loads a trained model and runs inference.
# Clear, explicit, and traceable.

import argparse
import torch
from typing import Any, List

from model import MyModel
from data import preprocess_input
from utils import load_checkpoint


def run_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    # Run inference - explicit forward pass.
    model.eval()
    
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
    
    return outputs


def main(args: argparse.Namespace) -> None:
    # Main inference function.
    #
    # Load model, process input, run prediction, return output.
    # Everything explicit and traceable.
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    print('Initializing model...')
    model = MyModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint_path}...')
    checkpoint_data = load_checkpoint(args.checkpoint_path, device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    print('Checkpoint loaded successfully')
    
    # Preprocess input
    print(f'\\nProcessing input from {args.input_path}...')
    inputs = preprocess_input(args.input_path)
    
    # Run inference
    print('Running inference...')
    outputs = run_inference(
        model=model,
        inputs=inputs,
        device=device
    )
    
    # Save or print results
    print('\\nPredictions:')
    print(outputs)
    
    if args.output_path:
        torch.save(outputs, args.output_path)
        print(f'\\nSaved predictions to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference with trained model'
    )
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension')
    
    # Input/Output arguments
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input data')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save predictions')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    main(args)
EOF

print_success "predict.py created"

# Create module stubs following the ML Research Framework
print_status "Creating module stubs..."

cat > data/__init__.py << 'EOF'
# Data module - dataset, preprocessing, and dataloader logic.
#
# This package contains multiple dataset implementations.
# Import the dataset you need in your training script.

from .example_dataset import ExampleDataset, create_dataloaders

__all__ = ['ExampleDataset', 'create_dataloaders']
EOF

cat > data/example_dataset.py << 'EOF'
# Example dataset implementation.
#
# Handles all data-related operations:
# - loading data from disk
# - preprocessing and transformations
# - creating dataloaders

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any


class ExampleDataset(Dataset):
    # Example dataset class - explicit and traceable.
    
    def __init__(self, data_path: str):
        # Initialize dataset.
        # TODO: Implement data loading
        self.data = None
        self.labels = None
    
    def __len__(self) -> int:
        # Return dataset size.
        # TODO: Implement
        return 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get a single item.
        # TODO: Implement
        pass


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    # Create train and validation dataloaders.
    #
    # Returns explicit DataLoader objects - no hidden magic.
    # TODO: Implement dataloader creation
    # Example:
    # train_dataset = ExampleDataset(f'{data_path}/train')
    # val_dataset = ExampleDataset(f'{data_path}/val')
    # 
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers
    # )
    # 
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers
    # )
    # 
    # return train_loader, val_loader
    pass


def preprocess_input(input_path: str) -> torch.Tensor:
    # Preprocess input for inference.
    # TODO: Implement preprocessing
    pass
EOF

cat > models/__init__.py << 'EOF'
# Models module - defines model architectures and forward computation.
#
# This package contains multiple model implementations.
# Import the model you need in your training script.

from .example_model import ExampleModel

__all__ = ['ExampleModel']
EOF

cat > models/example_model.py << 'EOF'
# Example model implementation.
#
# Should contain:
# - neural network modules
# - layers and blocks
# - forward pass
# - model wrapper class

import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    # Model architecture definition.
    #
    # Keep close to the research idea - explicit layers and forward pass.
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10
    ):
        super(ExampleModel, self).__init__()
        
        # Define layers explicitly
        # TODO: Implement your architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass - explicit and traceable.
        #
        # No hidden logic, just the forward computation.
        # TODO: Implement forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
EOF

cat > losses.py << 'EOF'
# Losses module - defines loss functions.
#
# For projects with custom or multiple losses, define them here.
# Standard losses (CrossEntropy, MSE) can be used directly from PyTorch.

import torch
import torch.nn as nn


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # Compute loss - explicit and traceable.
    #
    # For simple cases, this might just wrap nn.CrossEntropyLoss()
    # For complex cases (e.g., BERT with MLM + NSP), implement here.
    # TODO: Implement loss computation
    # Example:
    # criterion = nn.CrossEntropyLoss()
    # return criterion(outputs, targets)
    pass


class CustomLoss(nn.Module):
    # Custom loss function.
    #
    # Use this for research-specific losses that need special logic.
    
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Compute custom loss.
        # TODO: Implement custom loss logic
        pass
EOF

cat > metrics.py << 'EOF'
# Metrics module - contains reusable evaluation metrics.
#
# Define all evaluation metrics here:
# - accuracy
# - precision, recall, F1
# - custom research metrics

import torch
from typing import Dict


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    # Compute all evaluation metrics.
    #
    # Returns a dictionary of metric name -> value.
    # Everything explicit and traceable.
    metrics = {}
    
    # TODO: Implement metrics
    # Example:
    # predictions = torch.argmax(outputs, dim=1)
    # correct = (predictions == targets).sum().item()
    # total = targets.size(0)
    # metrics['accuracy'] = correct / total
    
    return metrics


def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> float:
    # Compute accuracy - explicit calculation.
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total
EOF

cat > utils.py << 'EOF'
# Utils module - generic and reusable helper functions.
#
# Contains project-wide helpers:
# - seed setting
# - logging setup
# - checkpoint save/load
# - other general-purpose utilities

import torch
import random
import numpy as np
import os
from typing import Dict, Any


def set_seed(seed: int = 42) -> None:
    # Set random seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> None:
    # Setup logging directory.
    os.makedirs(log_dir, exist_ok=True)
    print(f'Logging to {log_dir}')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str
) -> None:
    # Save model checkpoint - explicit save.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(
    path: str,
    device: torch.device
) -> Dict[str, Any]:
    # Load model checkpoint - explicit load.
    checkpoint = torch.load(path, map_location=device)
    return checkpoint
EOF

cat > config.py << 'EOF'
# Config module - defines configuration schema and loading logic.
#
# For projects that use config files (YAML, JSON), define loading here.
# Provides configuration object to the rest of the codebase.

import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    # Load configuration from YAML file.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Config:
    # Configuration class for experiment settings.
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Initialize config from dictionary.
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    @classmethod
    def from_file(cls, config_path: str):
        # Create config from YAML file.
        config_dict = load_config(config_path)
        return cls(config_dict)
EOF

print_success "All module stubs created"

# Create .gitignore
print_status "Creating .gitignore..."
cat > .gitignore << 'EOF'
# UV / Python
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
*.egg
.pytest_cache/
.coverage
htmlcov/
dist/
build/

# Project specific - outputs directory
outputs/
*.pth
*.pt
*.log

# Data
data/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
notebooks/*.ipynb

# Environment
.env
.envrc

# Configs (optionally ignore local configs)
# configs/local*.yaml
EOF

print_success ".gitignore created"

# Add dependencies
print_status "Adding core ML dependencies..."
uv add torch numpy pandas scikit-learn matplotlib tqdm pyyaml

print_status "Adding development dependencies..."
uv add --dev pytest black ruff jupyter ipython

# Sync environment
print_status "Syncing environment (this may take a moment)..."
uv sync

print_success "Environment synced successfully"

# Create README following ML Research Framework
print_status "Creating README.md..."
cat > README.md << EOF
# $PROJECT_NAME

A machine learning research project structured for clarity, reproducibility, and ease of understanding.

## Philosophy

This project follows the **ML Research Framework** principles:

- **What You See Is What You Get**: Code is explicit and traceable. If something breaks, it's our error, not framework magic.
- **Explicit Pipelines over Magic**: The training flow is visible from top to bottom without hidden callbacks or hooks.
- **Code Close to Research**: Implementation maps naturally to the research concepts.
- **Single Responsibility**: Each module focuses on one thing and does it well.

## Setup

### Prerequisites

- UV installed: \`curl -LsSf https://astral.sh/uv/install.sh | sh\`

### Installation

\`\`\`bash
# Clone the repository
git clone <your-repo-url>
cd $PROJECT_NAME

# Sync environment (installs all dependencies)
uv sync
\`\`\`

## Project Structure

Following the ML Research Framework:

\`\`\`
$PROJECT_NAME/
├── train.py          # Training orchestration - the main story
├── evaluate.py       # Independent checkpoint evaluation
├── predict.py        # Inference on new inputs
│
├── data/             # Dataset implementations
│   ├── __init__.py   # Package initialization
│   └── example_dataset.py  # Example dataset
├── models/           # Model architectures
│   ├── __init__.py   # Package initialization
│   └── example_model.py    # Example model
├── losses.py         # Loss function definitions
├── metrics.py        # Evaluation metrics
├── config.py         # Configuration schema
├── utils.py          # Generic helper functions
│
├── configs/          # Experiment configurations
├── tests/            # Unit tests
├── outputs/          # All outputs
│   ├── checkpoints/  # Model checkpoints
│   ├── logs/         # Training logs
│   └── predictions/  # Prediction outputs
├── notebooks/        # Jupyter notebooks
│
└── pyproject.toml    # Project configuration
\`\`\`

### Core Files and Their Roles

| File/Directory | Responsibility |
|------|----------------|
| \`train.py\` | Orchestrates training, validation, experiment flow |
| \`evaluate.py\` | Evaluates checkpoints independently |
| \`predict.py\` | Runs inference on new inputs |
| \`data/\` | Dataset implementations, preprocessing, dataloader creation |
| \`models/\` | Model architecture definitions and forward pass |
| \`losses.py\` | Loss function definitions |
| \`metrics.py\` | Evaluation metrics |
| \`utils.py\` | Generic helpers (seed, logging, checkpoints) |

**Key Principle**: Top-level scripts (\`train.py\`, \`evaluate.py\`, \`predict.py\`) are the only files that combine ingredients. Other modules don't import each other unnecessarily.

## Usage

### Training

\`\`\`bash
uv run train.py \\
    --data_path ./data \\
    --epochs 100 \\
    --batch_size 32 \\
    --learning_rate 0.001 \\
    --checkpoint_dir ./outputs/checkpoints \\
    --log_dir ./outputs/logs
\`\`\`

### Evaluation

\`\`\`bash
uv run evaluate.py \\
    --data_path ./data \\
    --checkpoint_path ./outputs/checkpoints/best_model.pth
\`\`\`

### Inference/Prediction

\`\`\`bash
uv run predict.py \\
    --input_path ./data/test_sample.pt \\
    --checkpoint_path ./outputs/checkpoints/best_model.pth \\
    --output_path ./outputs/predictions/results.pt
\`\`\`

## Development

### Adding Dependencies

\`\`\`bash
# Add a package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>
\`\`\`

### Running Tests

\`\`\`bash
uv run pytest tests/
\`\`\`

### Code Formatting

\`\`\`bash
# Format code
uv run black .

# Check linting
uv run ruff check .
\`\`\`

## Design Principles

### Explicit is Better Than Implicit

The training loop in \`train.py\` is written step-by-step:

\`\`\`python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_metrics = validate(...)
    
    # Check early stopping explicitly
    should_stop, patience_counter = check_early_stopping(...)
    if should_stop:
        print(f'Early stopping at epoch {epoch}')
        break
\`\`\`

No hidden callbacks, no magic hooks - just readable Python.

### Easy to Debug

When something breaks, you can trace exactly where:
- Data loading happens in \`data/\`
- Model forward pass is in \`models/\`
- Loss computation is in \`losses.py\`
- Training loop is in \`train.py\`

Everything is inspectable and step-by-step.

## License

[Your License Here]

## References

This project structure follows the [ML Research Project Framework](https://github.com/your-username/ml-research-framework).
EOF

print_success "README.md created"

# Create example config file
print_status "Creating example config file..."
cat > configs/example.yaml << 'EOF'
# Example experiment configuration
# Load with: config = Config.from_file('configs/example.yaml')

# Data
data_path: "./data"
batch_size: 32
num_workers: 4

# Model
input_dim: 128
hidden_dim: 256
output_dim: 10

# Training
epochs: 100
learning_rate: 0.001
weight_decay: 0.0
patience: 10

# Paths
checkpoint_dir: "./outputs/checkpoints"
log_dir: "./outputs/logs"

# Reproducibility
seed: 42
EOF

print_success "Example config created"

# Print summary
echo ""
echo "=========================================="
print_success "Project setup complete!"
echo "=========================================="
echo ""
echo "Project: $PROJECT_NAME"
echo "Location: $(pwd)"
echo ""
echo "Framework: ML Research Project Framework"
echo "  - Explicit and readable code"
echo "  - Single responsibility modules"
echo "  - Easy to debug and modify"
echo ""
echo "Next steps:"
echo "  1. Implement your modules:"
echo "     - data/ (dataset implementations)"
echo "     - models/ (model architectures)"
echo "     - losses.py (loss functions)"
echo "     - metrics.py (evaluation metrics)"
echo ""
echo "  2. Run training:"
echo "     uv run train.py --data_path ./data --epochs 10"
echo ""
echo "  3. Evaluate model:"
echo "     uv run evaluate.py --data_path ./data --checkpoint_path ./outputs/checkpoints/best_model.pth"
echo ""
echo "Installed packages:"
uv pip list | head -15
echo "  ... and more"
echo ""
print_status "Happy researching! 🔬"
