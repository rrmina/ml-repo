# UV Setup Tutorial: From Zero to Running Project

This tutorial walks you through setting up a machine learning project using UV, the fast Python package manager. Follow these steps in order.

## Prerequisites

- A computer running macOS, Linux, or Windows
- Terminal/command line access
- Internet connection

---

## Part 1: Installing UV

### Option 1: Install via Shell Script (Recommended for macOS/Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Option 2: Install via PowerShell (Windows)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Option 3: Install via pip

```bash
pip install uv
```

### Verify Installation

```bash
uv --version
```

You should see output like: `uv 0.x.x`

---

## Part 2: Creating a New Project

### Step 1: Create Project Directory

```bash
# Navigate to where you want your project
cd ~/Desktop

# Initialize a new UV project
uv init my_ml_project

# Enter the project directory
cd my_ml_project
```

This creates:
```
my_ml_project/
├── pyproject.toml   # Project configuration
├── hello.py         # Sample file (can delete)
└── README.md        # Project readme
```

### Step 2: Specify Python Version

Create a `.python-version` file to tell UV which Python version to use:

```bash
echo "3.12" > .python-version
```

UV will automatically download and use Python 3.12 if you don't have it installed.

---

## Part 3: Setting Up Project Structure

### Step 3: Create Project Files

```bash
# Create the main Python modules
touch data.py
touch models.py
touch train.py
touch eval.py
touch main.py
touch inference.py

# Create directories
mkdir -p data
mkdir -p checkpoints

# Optional: Remove the sample hello.py file
rm hello.py
```

Your structure should now look like:
```
my_ml_project/
├── .python-version
├── pyproject.toml
├── data.py
├── models.py
├── train.py
├── eval.py
├── main.py
├── inference.py
├── data/
├── checkpoints/
└── README.md
```

---

## Part 4: Adding Dependencies

### Step 4: Add Core ML Dependencies

UV makes adding packages incredibly fast:

```bash
# Add machine learning packages
uv add torch
uv add numpy
uv add pandas
uv add scikit-learn
uv add matplotlib
uv add tqdm
```

**What's happening?**
- UV downloads and installs packages (10-100x faster than pip)
- Updates `pyproject.toml` with the new dependencies
- Creates/updates `uv.lock` with exact versions for reproducibility

### Step 5: Add Development Dependencies

Add tools for development (testing, formatting, etc.):

```bash
uv add --dev pytest
uv add --dev black
uv add --dev ruff
uv add --dev jupyter
uv add --dev ipython
```

The `--dev` flag marks these as development-only dependencies.

### Step 6: View Your Dependencies

Check what's in your `pyproject.toml`:

```bash
cat pyproject.toml
```

You should see something like:
```toml
[project]
name = "my-ml-project"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.x.x",
    "numpy>=1.x.x",
    "pandas>=2.x.x",
    # ... more packages
]

[project.optional-dependencies]
dev = [
    "pytest>=7.x.x",
    "black>=23.x.x",
    # ... more dev tools
]
```

---

## Part 5: Installing and Syncing Environment

### Step 7: Sync Your Environment

This installs all dependencies from the lock file:

```bash
uv sync
```

**What does this do?**
- Creates a virtual environment (`.venv/`)
- Installs all packages from `uv.lock`
- Ensures deterministic, reproducible installs

### Step 8: Verify Installation

Check that packages are installed:

```bash
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

You should see the PyTorch version printed.

---

## Part 6: Writing Your First Code

### Step 9: Create a Simple Example in main.py

```bash
# Open main.py in your editor (or use cat/echo)
cat > main.py << 'EOF'
import argparse
import torch
import numpy as np

def main(args):
    print(f"Python ML Project")
    print(f"Mode: {args.mode}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if args.mode == "test":
        # Create a simple tensor
        x = torch.randn(3, 3)
        print(f"\nSample tensor:\n{x}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test",
                       choices=["test", "train", "eval"])
    args = parser.parse_args()
    main(args)
EOF
```

---

## Part 7: Running Your Project

### Step 10: Run Your Script with UV

```bash
uv run main.py --mode test
```

**Output should look like:**
```
Python ML Project
Mode: test
PyTorch version: 2.x.x
NumPy version: 1.x.x
CUDA available: False

Sample tensor:
tensor([[ 0.1234, -0.5678,  0.9012],
        [ 0.3456, -0.7890,  0.1234],
        [-0.5678,  0.9012, -0.3456]])
```

### Understanding `uv run`

```bash
# These are equivalent:
uv run main.py              # UV manages the environment
uv run python main.py       # Explicit python call

# You can also activate the environment manually:
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Then run directly:
python main.py
```

---

## Part 8: Working with UV Daily

### Adding New Packages

```bash
# Add a new dependency
uv add requests

# Add multiple packages
uv add seaborn plotly wandb

# Add with version constraint
uv add "torch>=2.0,<3.0"
```

### Removing Packages

```bash
uv remove requests
```

### Updating Packages

```bash
# Update all packages
uv lock --upgrade

# Update specific package
uv lock --upgrade-package torch

# Sync after updating
uv sync
```

### Running Commands

```bash
# Run Python scripts
uv run python script.py
uv run script.py

# Run Python REPL
uv run python

# Run installed tools
uv run pytest
uv run black .
uv run jupyter notebook
```

### Listing Packages

```bash
# List all installed packages
uv pip list

# Show package details
uv pip show torch
```

---

## Part 9: Sharing Your Project

### Step 11: Share with Others

When sharing your project with collaborators:

1. **Include in version control (git):**
   ```bash
   git add pyproject.toml uv.lock .python-version
   git commit -m "Add project configuration"
   ```

2. **Do NOT commit:**
   - `.venv/` (virtual environment)
   - `__pycache__/` (Python cache)
   - `.pytest_cache/` (pytest cache)

3. **Create a .gitignore file:**
   ```bash
   cat > .gitignore << 'EOF'
   # UV / Python
   .venv/
   __pycache__/
   *.pyc
   *.pyo
   *.egg-info/
   .pytest_cache/
   
   # Project specific
   checkpoints/*.pth
   data/*.csv
   *.log
   
   # IDE
   .vscode/
   .idea/
   *.swp
   EOF
   ```

### Step 12: How Others Can Set Up Your Project

Someone cloning your project just needs to:

```bash
# Clone the repository
git clone <your-repo-url>
cd my_ml_project

# Install UV (if they don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync environment (installs everything)
uv sync

# Run the project
uv run main.py --mode test
```

That's it! UV handles:
- Installing the correct Python version (3.11)
- Creating the virtual environment
- Installing exact package versions from `uv.lock`

---

## Part 10: Advanced UV Features

### Working with Multiple Python Versions

```bash
# Use a specific Python version for one command
uv run --python 3.10 main.py

# Change project Python version
echo "3.11" > .python-version
uv sync  # Recreates environment with Python 3.11
```

### Creating Scripts in pyproject.toml

Add shortcuts for common commands:

```toml
[project.scripts]
train = "my_ml_project.main:train"
eval = "my_ml_project.main:eval"

[project.optional-dependencies]
dev = [...]
```

Then run:
```bash
uv run train
uv run eval
```

### Using UV with Jupyter Notebooks

```bash
# Install Jupyter
uv add --dev jupyter ipykernel

# Launch Jupyter
uv run jupyter notebook

# Or use JupyterLab
uv add --dev jupyterlab
uv run jupyter lab
```

---

## Troubleshooting

### Issue: "uv: command not found"

**Solution:** Add UV to your PATH:
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Issue: Python version not found

**Solution:** UV will auto-download Python. If it fails:
```bash
# Manually specify Python
uv venv --python 3.12

# Or use system Python
uv venv --python python3.12
```

### Issue: Package conflicts

**Solution:** UV's resolver is smart, but if you hit issues:
```bash
# Clear cache and retry
uv cache clean
uv sync

# Check what's causing conflicts
uv pip tree
```

### Issue: Slow on first run

**First time UV downloads packages, it caches them**. Subsequent installs are much faster.

---

## Quick Reference Card

```bash
# Project Setup
uv init my_project          # Create new project
uv sync                     # Install all dependencies

# Package Management
uv add <package>            # Add dependency
uv add --dev <package>      # Add dev dependency
uv remove <package>         # Remove dependency
uv lock --upgrade           # Update all packages

# Running Code
uv run <script.py>          # Run Python script
uv run python               # Start Python REPL
uv run pytest               # Run tests

# Information
uv pip list                 # List installed packages
uv pip show <package>       # Show package details
uv --version               # UV version

# Environment
source .venv/bin/activate   # Activate venv (macOS/Linux)
.venv\Scripts\activate      # Activate venv (Windows)
deactivate                  # Deactivate venv
```

---

## Next Steps

Now that you have UV set up, you can:

1. ✅ Create your data loading code in `data.py`
2. ✅ Define your model architecture in `models.py`
3. ✅ Implement training logic in `train.py`
4. ✅ Add evaluation metrics in `eval.py`
5. ✅ Orchestrate everything in `main.py`
6. ✅ Create inference demos in `inference.py`

Refer to the main project framework documentation for implementation details of each component.

---

## Why UV?

- **⚡ Speed:** 10-100x faster than pip
- **🔒 Reliability:** Lock files ensure reproducible installs
- **🎯 Modern:** Uses pyproject.toml (Python standard)
- **🐍 Smart:** Automatic Python version management
- **🔧 Complete:** Replaces pip, pip-tools, virtualenv, and more

Happy coding with UV! 🚀
