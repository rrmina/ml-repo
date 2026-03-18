# ML Project Setup Script using UV (PowerShell)
# This script initializes a complete ML project structure with UV

# Error handling
$ErrorActionPreference = "Stop"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Get project name from user or use default
$PROJECT_NAME = $args[0]
if (-not $PROJECT_NAME) {
    $PROJECT_NAME = Read-Host "Enter project name (default: my_ml_project)"
    if ([string]::IsNullOrWhiteSpace($PROJECT_NAME)) {
        $PROJECT_NAME = "my_ml_project"
    }
}

Write-Status "Setting up ML project: $PROJECT_NAME"

# Check if UV is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warning "UV is not installed. Installing now..."
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Success "UV installed successfully"
} else {
    Write-Success "UV is already installed ($(uv --version))"
}

# Initialize UV project
Write-Status "Initializing UV project..."
uv init $PROJECT_NAME
Set-Location $PROJECT_NAME

# Set Python version
Write-Status "Setting Python version to 3.12..."
"3.12" | Out-File -FilePath ".python-version" -Encoding utf8 -NoNewline

# Create project structure
Write-Status "Creating project structure..."

# Create main Python files
New-Item -ItemType File -Name "data.py" -Force | Out-Null
New-Item -ItemType File -Name "models.py" -Force | Out-Null
New-Item -ItemType File -Name "train.py" -Force | Out-Null
New-Item -ItemType File -Name "eval.py" -Force | Out-Null
New-Item -ItemType File -Name "main.py" -Force | Out-Null
New-Item -ItemType File -Name "inference.py" -Force | Out-Null

# Create directories
New-Item -ItemType Directory -Name "data" -Force | Out-Null
New-Item -ItemType Directory -Name "checkpoints" -Force | Out-Null

# Remove default hello.py if it exists
if (Test-Path "hello.py") {
    Remove-Item "hello.py"
}

Write-Success "Project structure created"

# Create main.py with template
Write-Status "Creating main.py template..."
@'
import argparse
import torch
from typing import Any

# Import all project modules
from data import load_dataset, preprocess_data, create_dataloaders
from models import MyModel
from train import train_model
from eval import evaluate_model
from inference import run_inference


# Main function that orchestrates the entire workflow.
def main(
    args: argparse.Namespace
) -> None:
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.mode == 'train':
        print('Training mode...')
        
        # TODO: Implement training workflow
        # Example:
        # raw_data = load_dataset(args.data_path)
        # processed_data = preprocess_data(raw_data)
        # train_loader, val_loader = create_dataloaders(processed_data, args.batch_size)
        # 
        # model = MyModel(
        #     input_dim=args.input_dim,
        #     hidden_dim=args.hidden_dim,
        #     output_dim=args.output_dim
        # ).to(device)
        # 
        # train_model(
        #     model=model,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     epochs=args.epochs,
        #     learning_rate=args.lr,
        #     device=device,
        #     save_path=args.checkpoint_path
        # )
        
    elif args.mode == 'eval':
        print('Evaluation mode...')
        
        # TODO: Implement evaluation workflow
        # Example:
        # model = MyModel().to(device)
        # model.load_state_dict(torch.load(args.checkpoint_path))
        # model.eval()
        # 
        # raw_data = load_dataset(args.data_path)
        # processed_data = preprocess_data(raw_data)
        # _, test_loader = create_dataloaders(processed_data)
        # 
        # results = evaluate_model(
        #     model=model,
        #     test_loader=test_loader,
        #     device=device
        # )
        # print(f'Evaluation Results: {results}')
        
    elif args.mode == 'inference':
        print('Inference mode...')
        
        # TODO: Implement inference workflow
        # Example:
        # result = run_inference(
        #     checkpoint_path=args.checkpoint_path,
        #     input_data=args.input,
        #     device=device
        # )
        # print(f'Inference Result: {result}')
    
    else:
        print(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Project Main Script')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'eval', 'inference'],
                       help='Mode: train, eval, or inference')
    
    parser.add_argument('--data_path', type=str, default='./data/dataset.csv',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--checkpoint_path', type=str, 
                       default='./checkpoints/model.pth',
                       help='Path to save/load model checkpoint')
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input for inference mode')
    
    args = parser.parse_args()
    main(args)
'@ | Out-File -FilePath "main.py" -Encoding utf8

# Create stub files for other modules
Write-Status "Creating module stubs..."

@'
# Data loading and preprocessing module.
from typing import Any, Tuple

# Load dataset from file.
def load_dataset(
    data_path: str
) -> Any:
    
    # TODO: Implement data loading
    pass

# Preprocess raw data.
def preprocess_data(
    raw_data: Any
) -> Any:
    
    # TODO: Implement preprocessing
    pass

# Create data loaders for training/validation.
def create_dataloaders(
    processed_data: Any,
    batch_size: int = 32
) -> Tuple[Any, Any]:
    
    # TODO: Implement dataloader creation
    pass
'@ | Out-File -FilePath "data.py" -Encoding utf8

@'
# Model architecture definitions.
import torch
import torch.nn as nn
from typing import Optional

# Neural network model.
class MyModel(nn.Module):
    def __init__(self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10
    ) -> None:
        
        super(MyModel, self).__init__()
        
        # TODO: Define model architecture
        pass
    
    # Forward pass.
    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        # TODO: Implement forward pass
        pass
'@ | Out-File -FilePath "models.py" -Encoding utf8

@'
# Training logic and utilities.
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

# Train the model.
def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    save_path: str
) -> None:
    
    # TODO: Implement training loop
    # Sample implementation:
    
    # # Set up optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    # 
    # # Training loop
    # for epoch in range(epochs):
    #     model.train()
    #     train_loss = 0.0
    #     
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #     
    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         ...
    #     
    #     print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}')
    # 
    # # Save model
    # torch.save(model.state_dict(), save_path)
    pass
'@ | Out-File -FilePath "train.py" -Encoding utf8

@'
# Evaluation and metrics.
import torch
import torch.nn as nn
from typing import Any, Dict

# Evaluate model performance.
def evaluate_model(
    model: nn.Module,
    test_loader: Any,
    device: torch.device
) -> Dict[str, float]:
    
    # TODO: Implement evaluation
    pass
'@ | Out-File -FilePath "eval.py" -Encoding utf8

@'
# Inference and prediction utilities.
import torch
from typing import Any, Optional

# Run inference on new data.
def run_inference(
    checkpoint_path: str,
    input_data: Any,
    device: torch.device
) -> Any:
    
    # TODO: Implement inference
    pass

if __name__ == '__main__':
    print('Inference demo')
    
    # TODO: Add demo code
'@ | Out-File -FilePath "inference.py" -Encoding utf8

Write-Success "Module stubs created"

# Create .gitignore
Write-Status "Creating .gitignore..."
@'
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

# Project specific
checkpoints/*.pth
checkpoints/*.pt
data/*.csv
data/*.json
data/*.pkl
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.envrc
'@ | Out-File -FilePath ".gitignore" -Encoding utf8

Write-Success ".gitignore created"

# Add dependencies
Write-Status "Adding core ML dependencies..."
uv add torch numpy pandas scikit-learn matplotlib tqdm

Write-Status "Adding development dependencies..."
uv add --dev pytest black ruff jupyter ipython

# Sync environment
Write-Status "Syncing environment (this may take a moment)..."
uv sync

Write-Success "Environment synced successfully"

# Create README
Write-Status "Creating README.md..."
@"
# $PROJECT_NAME

A machine learning project using UV for fast dependency management.

## Setup

This project was initialized with the ML project setup script.

### Prerequisites

- UV installed: ``powershell -c "irm https://astral.sh/uv/install.ps1 | iex"``

### Installation

``````bash
# Clone the repository
git clone <your-repo-url>
cd $PROJECT_NAME

# Sync environment (installs all dependencies)
uv sync
``````

## Project Structure

``````
$PROJECT_NAME/
├── data.py           # Data loading and preprocessing
├── models.py         # Model architecture definitions
├── train.py          # Training logic
├── eval.py           # Evaluation metrics
├── main.py           # Main orchestration script
├── inference.py      # Inference and prediction
├── data/             # Dataset directory
├── checkpoints/      # Model checkpoints
└── pyproject.toml    # Project configuration
``````

## Usage

### Training

``````bash
uv run main.py --mode train --epochs 10 --lr 0.001
``````

### Evaluation

``````bash
uv run main.py --mode eval --checkpoint_path ./checkpoints/model.pth
``````

### Inference

``````bash
uv run main.py --mode inference --checkpoint_path ./checkpoints/model.pth
``````

## Development

### Adding Dependencies

``````bash
uv add <package-name>
``````

### Running Tests

``````bash
uv run pytest
``````

### Code Formatting

``````bash
uv run black .
uv run ruff check .
``````

## License

[Your License Here]
"@ | Out-File -FilePath "README.md" -Encoding utf8

Write-Success "README.md created"

# Print summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Success "Project setup complete!"
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Project: $PROJECT_NAME"
Write-Host "Location: $(Get-Location)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. cd $PROJECT_NAME (if not already there)"
Write-Host "  2. Implement your modules (data.py, models.py, etc.)"
Write-Host "  3. Run your project: uv run main.py --mode train"
Write-Host ""
Write-Host "Installed packages:"
uv pip list | Select-Object -First 15
Write-Host "  ... and more"
Write-Host ""
Write-Status "Happy coding! 🚀"
