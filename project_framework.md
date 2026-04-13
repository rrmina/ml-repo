# ML Research Project Codebase Framework

This document defines the recommended codebase structure for machine learning research projects.

## Goal

The goal is to ensure that projects are:

- easy to read
- easy to debug
- easy to modify
- easy to reproduce
- easy to hand-over to other researchers

This framework is intentionally designed for **research workflows**, not production systems.

It prioritizes:

- fast iteration
- transparent implementation
- experimental flexibility
- low abstraction overhead

## Design Philosophy

### 1. What You See Is What You Get

**Principle:**  
> If something breaks, we want the error to be our error, and not a side-effect of unnecessary abstraction or external framework complexity.

The codebase should be written in a way that makes behaviour explicit and easy to trace.

An audience should be able to understand:

- where data is loaded
- how it is transformed
- how the model works
- how training is performed
- how loss is computed
- how outputs are evaluated

The implementations should not rely on excessive abstraction, hidden framework logic, or deeply nested modules.

We prefer code that is:

- explicit
- inspectable
- step-by-step
- easy-to-debug

over code that is:

- overly clever
- highly abstracted
- framework heavy
- difficult to trace during failure

### 2. Prefer Explicit Pipelines over Magic!

**Principle:**  
> The code should be understandable by reading the project's own code, not by reading framework internals.

Abstraction should not be introduced just to make the code look cleaner or more sophisticated, as too much abstraction creates hidden assumptions and debugging difficulty.

The main training and evaluation flows should be visible from top to bottom. A new research member should be able to read `main.py` or `train.py` and understand the experiment without needing to mentally trace many deep and hidden layers of indirection.

**Preferred style:**

```python
import torch

# Load trainloader, model, optimizer, and loss function
trainloader = torch.utils.data.DataLoader(...)
model = Seq2Seq(...)
optimizer = torch.optim.Adam(...)
loss_fn = torch.nn.CrossEntropyLoss(...)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = evaluate(...)
    
    # Check for early stopping
    early_stopping_condition_met = \
        check_early_stopping(val_loss, best_val_loss, patience)
    
    if early_stopping_condition_met:
        print(f'Early stopping at epoch {epoch}')
        break
```

Avoid code design where the actual behavior is hidden behind:

- callback stacks
- lifecycle hooks

### 3. Keep the Code Close to the Math / Research Idea

**Principle:**  
> The code should reflect the actual research idea, not bury it under software architecture patterns.

For research projects, the code should map naturally to the conceptual components of the method.

For example, if the research involves:

- an encoder
- a decoder
- a ranking loss
- negative sampler
- a retrieval stage

then those concepts should appear clearly in the codebase, preferably in their **own module**.

### 4. Each ML Ingredient Has Its Own Role

**Principle:**  
> No module should do more than its intended role. Each ingredient contributes a single responsibility to the experiment recipe!

This framework is organized around the principle that each component should focus on its own responsibility.

| File/Directory | Responsibility |
|------|----------------|
| `train.py` | orchestrates training, validation, and experiment flow |
| `data/` | dataset, preprocessing, and dataloader logic for multiple datasets |
| `models/` | defines model architectures and forward computation for multiple models |
| `losses.py` | defines the loss functions |
| `predict.py` | handles inference on new inputs |
| `evaluate.py` | handles evaluation of checkpoints independently |
| `metrics.py` | contains reusable evaluation metrics |
| `utils.py` / Fabric | contains generic and reusable 
helper functions not specific to any other component (e.g., loading and saving functions, logging, etc) |

**Practical Implications:**

- Top-level scripts (`train.py`, `evaluate.py`, `predict.py`) are the only files that combine these ingredients to execute the experimental logic.
- Other modules should not import each other unnecessarily.


--------------------

## Non-Goals

To keep the framework aligned with the research needs, the following are not primary goals:

- optimizing for production deployment from day one
- introducting software architecture patterns before they are necessary
- reducing line count at the cost of readability and understandability

> Shorter code is not automatically better code, and at the same time, more abstract code is not necessarily better code. 

--------------------

## Recommended Project Structure

```text
project/
│
├── train.py
├── evaluate.py
├── predict.py
│
├── data/
│   ├── __init__.py
│   ├── dataset1.py
│   └── dataset2.py
│
├── models/
│   ├── __init__.py
│   ├── model1.py
│   └── model2.py
│
├── losses.py
├── metrics.py
├── config.py
├── utils.py
│
├── configs/
│   ├── base.yaml
│   └── experiment_1.yaml
│
├── tests/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore

```

This is the recommended default structure for research projects.

**Not all projects will require every file or folder**, but this should be the baseline starting point of the project. 

For example, a standard Transformer that only uses log-likelihood loss does not require a losses.py file. However, for a model like BERT, which uses both Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) losses, it is highly recommended to define these in a separate `losses.py` file to be imported into `train.py`.

### When to Use Directories vs. Single Files

**For `data/` and `models/`:**

- **Use a directory** when your project involves:
  - Multiple different datasets (e.g., ImageNet, CIFAR-10, custom datasets)
  - Multiple different model architectures (e.g., Transformer, LSTM, CNN)
  - Shared components across datasets or models (e.g., common transforms, shared layers)
  
- **Use a single file** (`data.py` or `model.py`) when:
  - Your project uses only one dataset
  - Your project implements only one model architecture
  - The code complexity is low enough to maintain in a single file

**Migration path:** It's perfectly acceptable to start with single files (`data.py`, `model.py`) and migrate to directories as your project grows and requires multiple implementations.

## Core File and Responsibilities

### `train.py`

#### Purpose

> `train.py` should read like the main story of the experiment

#### Responsibilities

* load configuration, ideally via argparse or config.py
* initialize dataset/dataloaders
* initialize model
* initialize optimizer, scheduler, loss
* execute training loop
* trigger validation
* save checkpoints
* log results

#### Should not contain

* model architecture definisitions
* dataset and preprocessing details
* reusable helpper function unrelated to orchestration

### `models/`

#### Purpose
> `models/` contains model architecture definitions and forward computation for multiple models

#### Responsibilities

* neural network modules
* layers and blocks
* forward pass
* model wrapper classes
* one file per model architecture (e.g., `transformer.py`, `lstm.py`, `cnn.py`)

#### Recommended Structure

```text
models/
├── __init__.py          # Import and expose model classes
├── transformer.py       # Transformer model implementation
├── lstm.py              # LSTM model implementation
└── components.py        # Shared model components (optional)
```

#### Should not contain

* optimizer setup
* checkpoint loading/saving
* evaluation metric
* training loops

#### Usage Example

```python
# In train.py
from models.transformer import TransformerModel
from models.lstm import LSTMModel

if config.model_type == 'transformer':
    model = TransformerModel(...)
elif config.model_type == 'lstm':
    model = LSTMModel(...)
```

### `data/`

#### Purpose
> `data/` contains data-related logic for multiple datasets

#### Responsibilities

* dataset class definitions
* caching of large datasets
* dataloader creation
* preprocessing
* all that transforms raw inputs into model-ready tensors
* one file per dataset type (e.g., `imagenet.py`, `cifar10.py`, `custom_dataset.py`)

#### Recommended Structure

```text
data/
├── __init__.py          # Import and expose dataset classes
├── imagenet.py          # ImageNet dataset implementation
├── cifar10.py           # CIFAR-10 dataset implementation
├── custom_dataset.py    # Custom dataset implementation
└── transforms.py        # Shared data transformations (optional)
```

#### Should not contain

* training loop logic
* model architecture
* experiment-specific reports

#### Usage Example

```python
# In train.py
from data.imagenet import ImageNetDataset
from data.cifar10 import CIFAR10Dataset

if config.dataset == 'imagenet':
    train_dataset = ImageNetDataset(...)
elif config.dataset == 'cifar10':
    train_dataset = CIFAR10Dataset(...)

train_loader = torch.utils.data.DataLoader(train_dataset, ...)
```

### `losses.py`

#### Purpose
> `losses.py` defines the loss functions

#### Responsibilities

* loss function definitions and loss computation logic
* any loss-specific helper functions

#### Should not contian

* model architecture
* training loop logic
* evaluation metric

### `evaluate.py`

#### Purpose
> `evaluate.py` runs evaluation on validation or test data

#### Responsibilities

* load trained checkpoint
* run model on validation/test set
* computer evaluation metrics
* generatee evaluation outputs and reports

### `predict.py` | `inference.py`

> `predict.py` runs inference using a trained model

#### Responsibilities

* load checkpoint config
* load trained checkpoint
* preprocess input data
* run prediction
* return or save outputs

### `metrics.py`

#### Purpose

> `metrics.py` defines reusable evaluation metrics

#### Responsibilities

* model performance metrics
* business metrics

#### Should not contain

* tensor-specific loss functions

### `utils.py` | Lightning Fabric

#### Purpose

> `utils.py` contains small, reusuable, project-wide helper functions

#### Responsibilities

* seed setting
* logging helpers
* checkpoint save and load wrappers
* other general-purpose file I/O
* anything not owned by any of the core ingredients

#### Should not contain

* preprocessing
* math

### `config.py`

#### Purpose

> `config.py` defines the configuration schema and loading logic

#### Responsibilities

* define configuration schema
* helper function for loading configs from file
* provide configuration object to the rest of the codebase

## Optional Folders
### `configs/`

* store experiment configurations

### `outputs/`

* store outputs

### `notebooks/`

* store notebooks

## Optional scripts

### run_experiment.sh

#### Purpose
> `run_experiment.sh` is a one-click script that runs the entire project and reproduces the results.

#### Responsibilities

* Set-up full experiment pipeline
* Call the appropriate project modules in sequence
* Ensure that experiments can be reproduced with minimal manual intervention

