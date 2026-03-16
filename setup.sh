#!/bin/bash

# ML Project Setup Script using UV
# This script initializes a complete ML project structure with UV

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
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "UV installed successfully"
else
    print_success "UV is already installed ($(uv --version))"
fi

# Initialize UV project
print_status "Initializing UV project..."
uv init "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Set Python version
print_status "Setting Python version to 3.11..."
echo "3.11" > .python-version

# Create project structure
print_status "Creating project structure..."

# Create main Python files
touch data.py
touch models.py
touch train.py
touch eval.py
touch main.py
touch inference.py

# Create directories
mkdir -p data
mkdir -p checkpoints

# Remove default hello.py if it exists
if [ -f "hello.py" ]; then
    rm hello.py
fi

print_success "Project structure created"

# Create main.py with template
print_status "Creating main.py template..."
cat > main.py << 'EOF'
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
        print("Training mode...")
        # TODO: Implement training workflow
        
    elif args.mode == 'eval':
        print("Evaluation mode...")
        # TODO: Implement evaluation workflow
        
    elif args.mode == 'inference':
        print("Inference mode...")
        # TODO: Implement inference workflow
    
    else:
        print(f"Unknown mode: {args.mode}")


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
EOF

# Create stub files for other modules
print_status "Creating module stubs..."

cat > data.py << 'EOF'
"""Data loading and preprocessing module."""

def load_dataset(data_path):
    """Load dataset from file."""
    # TODO: Implement data loading
    pass

def preprocess_data(raw_data):
    """Preprocess raw data."""
    # TODO: Implement preprocessing
    pass

def create_dataloaders(processed_data, batch_size=32):
    """Create data loaders for training/validation."""
    # TODO: Implement dataloader creation
    pass
EOF

cat > models.py << 'EOF'
"""Model architecture definitions."""
import torch.nn as nn

class MyModel(nn.Module):
    """Neural network model."""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super(MyModel, self).__init__()
        # TODO: Define model architecture
        pass
    
    def forward(self, x):
        """Forward pass."""
        # TODO: Implement forward pass
        pass
EOF

cat > train.py << 'EOF'
"""Training logic and utilities."""

def train_model(model, train_loader, val_loader, epochs, learning_rate, device, save_path):
    """Train the model."""
    # TODO: Implement training loop
    pass
EOF

cat > eval.py << 'EOF'
"""Evaluation and metrics."""

def evaluate_model(model, test_loader, device):
    """Evaluate model performance."""
    # TODO: Implement evaluation
    pass
EOF

cat > inference.py << 'EOF'
"""Inference and prediction utilities."""

def run_inference(checkpoint_path, input_data, device):
    """Run inference on new data."""
    # TODO: Implement inference
    pass

if __name__ == '__main__':
    print("Inference demo")
    # TODO: Add demo code
EOF

print_success "Module stubs created"

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
EOF

print_success ".gitignore created"

# Add dependencies
print_status "Adding core ML dependencies..."
uv add torch numpy pandas scikit-learn matplotlib tqdm

print_status "Adding development dependencies..."
uv add --dev pytest black ruff jupyter ipython

# Sync environment
print_status "Syncing environment (this may take a moment)..."
uv sync

print_success "Environment synced successfully"

# Create README
print_status "Creating README.md..."
cat > README.md << EOF
# $PROJECT_NAME

A machine learning project using UV for fast dependency management.

## Setup

This project was initialized with the ML project setup script.

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

\`\`\`
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
\`\`\`

## Usage

### Training

\`\`\`bash
uv run main.py --mode train --epochs 10 --lr 0.001
\`\`\`

### Evaluation

\`\`\`bash
uv run main.py --mode eval --checkpoint_path ./checkpoints/model.pth
\`\`\`

### Inference

\`\`\`bash
uv run main.py --mode inference --checkpoint_path ./checkpoints/model.pth
\`\`\`

## Development

### Adding Dependencies

\`\`\`bash
uv add <package-name>
\`\`\`

### Running Tests

\`\`\`bash
uv run pytest
\`\`\`

### Code Formatting

\`\`\`bash
uv run black .
uv run ruff check .
\`\`\`

## License

[Your License Here]
EOF

print_success "README.md created"

# Print summary
echo ""
echo "=========================================="
print_success "Project setup complete!"
echo "=========================================="
echo ""
echo "Project: $PROJECT_NAME"
echo "Location: $(pwd)"
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME (if not already there)"
echo "  2. Implement your modules (data.py, models.py, etc.)"
echo "  3. Run your project: uv run main.py --mode train"
echo ""
echo "Installed packages:"
uv pip list | head -15
echo "  ... and more"
echo ""
print_status "Happy coding! 🚀"
