# Setup Scripts Usage Guide

This directory contains automated setup scripts for quickly initializing an ML project with UV.

## Available Scripts

- **`setup.sh`** - Bash script for macOS/Linux
- **`setup.ps1`** - PowerShell script for Windows
- **`setup.bat`** - Batch script for Windows (basic version)

## Quick Start

### macOS / Linux

```bash
# Run the setup script
./setup.sh

# Or specify a project name
./setup.sh my_awesome_project
```

### Windows (PowerShell - Recommended)

```powershell
# Run the setup script
powershell -ExecutionPolicy Bypass -File setup.ps1

# Or specify a project name
powershell -ExecutionPolicy Bypass -File setup.ps1 my_awesome_project
```

### Windows (Command Prompt)

```cmd
setup.bat
```

**Note:** The .bat script provides basic setup only. Use PowerShell for full functionality.

## What the Scripts Do

The setup scripts automate the entire project initialization:

1. ✅ **Check/Install UV** - Verifies UV is installed, installs if missing
2. ✅ **Initialize Project** - Creates UV project with `uv init`
3. ✅ **Set Python Version** - Creates `.python-version` file (Python 3.11)
4. ✅ **Create Structure** - Creates all project files and directories:
   - `data.py`, `models.py`, `train.py`, `eval.py`, `main.py`, `inference.py`
   - `data/` and `checkpoints/` directories
5. ✅ **Generate Templates** - Creates starter code in each module
6. ✅ **Add Dependencies** - Installs ML packages:
   - Core: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`
   - Dev: `pytest`, `black`, `ruff`, `jupyter`, `ipython`
7. ✅ **Sync Environment** - Runs `uv sync` to install everything
8. ✅ **Create .gitignore** - Adds appropriate ignore patterns
9. ✅ **Create README** - Generates project README with usage instructions

## After Running the Script

Your project structure will look like:

```
my_project/
├── .python-version      # Python 3.11
├── .gitignore          # Git ignore rules
├── pyproject.toml      # UV dependencies
├── uv.lock            # Lock file
├── README.md          # Project documentation
├── main.py            # Main orchestration (with template code)
├── data.py            # Data loading (stub)
├── models.py          # Model definitions (stub)
├── train.py           # Training logic (stub)
├── eval.py            # Evaluation metrics (stub)
├── inference.py       # Inference utilities (stub)
├── data/              # Dataset directory
└── checkpoints/       # Model checkpoints
```

## Next Steps

1. **Navigate to your project:**
   ```bash
   cd my_project
   ```

2. **Verify installation:**
   ```bash
   uv run python -c "import torch; print(torch.__version__)"
   ```

3. **Implement your modules:**
   - Edit `data.py` to load your dataset
   - Define your model in `models.py`
   - Implement training logic in `train.py`
   - Add evaluation metrics in `eval.py`
   - Complete the workflow in `main.py`

4. **Run your project:**
   ```bash
   uv run main.py --mode train
   ```

## Script Features

### Interactive
- Prompts for project name if not provided
- Colored output for better readability
- Progress indicators for each step

### Robust
- Checks for UV installation before proceeding
- Handles errors gracefully
- Validates each step before continuing

### Complete
- Sets up everything you need to start coding
- No manual configuration required
- Ready-to-run template code

## Customization

You can modify the scripts to:
- Change Python version (edit the `.python-version` creation)
- Add/remove dependencies (edit the `uv add` commands)
- Customize directory structure (edit the `mkdir` commands)
- Modify template code (edit the heredoc sections)

## Troubleshooting

### UV Not Found After Installation

**Linux/macOS:**
```bash
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc
```

**Windows:**
Restart your terminal after UV installation.

### Permission Denied (Linux/macOS)

```bash
chmod +x setup.sh
```

### PowerShell Execution Policy (Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or run with bypass:
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

### Script Fails Midway

The scripts are idempotent - you can safely re-run them. They will:
- Skip UV installation if already present
- Overwrite files if they exist
- Continue from where it failed

## Manual Setup Alternative

If you prefer manual setup, follow these commands:

```bash
# Initialize project
uv init my_project
cd my_project

# Set Python version
echo "3.11" > .python-version

# Create files
touch data.py models.py train.py eval.py main.py inference.py
mkdir -p data checkpoints

# Add dependencies
uv add torch numpy pandas scikit-learn matplotlib tqdm
uv add --dev pytest black ruff jupyter ipython

# Sync environment
uv sync
```

## Examples

### Create a project named "sentiment_analysis"
```bash
./setup.sh sentiment_analysis
```

### Create a project in a specific location
```bash
cd ~/Projects
./setup.sh image_classifier
```

### Run and verify immediately
```bash
./setup.sh text_generator
cd text_generator
uv run python -c "import torch; print('Setup complete!')"
```

## Support

For issues with:
- **UV itself**: Visit [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **These scripts**: Check the inline comments in the script files
- **Project framework**: Refer to the other documentation files

---

Happy coding! 🚀
