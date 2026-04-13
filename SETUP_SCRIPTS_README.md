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
3. ✅ **Set Python Version** - Creates `.python-version` file (Python 3.12)
4. ✅ **Create Structure** - Creates all project files and directories:
   - `train.py`, `eval.py`, `main.py`, `inference.py`
   - `data/` and `models/` module directories with `__init__.py` files
   - `checkpoints/` directory
5. ✅ **Generate Templates** - Creates starter code in each module
   - Includes type annotations and proper formatting
   - `main.py` has commented example code showing how to use all imported functions
   - Each workflow mode (train/eval/inference) includes pseudocode examples
   - `train.py` includes a commented sample training loop (uncomment to use)
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
├── .python-version      # Python 3.12
├── .gitignore          # Git ignore rules
├── pyproject.toml      # UV dependencies
├── uv.lock            # Lock file
├── README.md          # Project documentation
├── main.py            # Main orchestration (with template code)
├── train.py           # Training logic (stub)
├── eval.py            # Evaluation metrics (stub)
├── inference.py       # Inference utilities (stub)
├── data/              # Dataset implementations
│   ├── __init__.py
│   └── example_dataset.py
├── models/            # Model architectures
│   ├── __init__.py
│   └── example_model.py
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
   - Edit `data/` to load your datasets
   - Define your models in `models/`
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
- Uses single quotes (') for Python strings following PEP 8 style
- Uses single-line comments (#) above functions/classes instead of docstrings
- Multi-line argument structure with type annotations
- Type hints for parameters and return types
- Blank line after function/method definitions before first comment/code
- Blank line before all TODO comments for better readability
- **Example workflow code** in `main.py` showing how to use all components

## Customization

You can modify the scripts to:
- Change Python version (edit the `.python-version` creation, currently 3.12)
- Add/remove dependencies (edit the `uv add` commands)
- Customize directory structure (edit the `mkdir` commands)
- Modify template code (edit the heredoc sections)

### Code Style

The generated Python code follows these conventions:
- **Single quotes** for strings: `print('Hello')`
- **Single-line comments** above functions/classes instead of docstrings:
  ```python
  # Load dataset from file.
  def load_dataset(
      data_path: str
  ) -> Any:
      
      # TODO: Implement data loading
      pass
  ```
- **Multi-line argument structure** with type annotations:
  ```python
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
  ```
- **Type hints** for all function parameters and return types
- **Multiple single-line comments** for longer descriptions
- **Blank line** after function/method definition before first comment/code:
  ```python
  def train_model(
      model: nn.Module,
      epochs: int
  ) -> None:
      
      # TODO: Implement training loop
      pass
  ```
- **Blank line before TODO comments** in code blocks:
  ```python
  if args.mode == 'train':
      print('Training mode...')
      
      # TODO: Implement training workflow
  ```

### Example Code in main.py

The generated `main.py` includes commented examples for each workflow mode:

**Training Mode Example:**
```python
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
```

**Evaluation Mode Example:**
```python
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
```

**Inference Mode Example:**
```python
# TODO: Implement inference workflow
# Example:
# result = run_inference(
#     checkpoint_path=args.checkpoint_path,
#     input_data=args.input,
#     device=device
# )
# print(f'Inference Result: {result}')
```

These examples show you exactly how to connect all the pieces together!

### Sample Training Implementation in train.py

The `train.py` includes a commented sample training loop template that you can uncomment and adapt:

```python
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
```

Simply uncomment the lines to use this template as a starting point for your training implementation!

## Troubleshooting

### UV Not Found After Installation

**Linux/macOS:**
```bash
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc
```

**Note:** UV might also install to `~/.cargo/bin` depending on your system.

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
echo "3.12" > .python-version

# Create files and directories
touch train.py eval.py main.py inference.py
mkdir -p data models checkpoints
touch data/__init__.py models/__init__.py

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
