# ML Ops Project

We will use uv to manage our virtual environment.

Install uv first 


Install dependencies with:
```bash
uv add -r requirements.txt
```

To activate the virtual environment, use:
```bash
source .venv/bin/activate
```

To deactivate the virtual environment, simply run:
```bash
deactivate
```

Create a freeze file to list dependencies with versions to reproduce the environment:
```bash
uv pip freeze > requirements-freeze.txt
```

To run the scripts of the project, use:
```bash
uv run src/prepare.py data/raw/ data/prepared
uv run src/train.py data/prepared/ model
uv run src/evaluate.py model/ data/prepared/
```

## Project Structure
```bash
> tree . -L 3
.
├── README.md
├── data
│   ├── prepared
│   │   ├── labels.json
│   │   ├── preview.png
│   │   ├── test
│   │   └── train
│   └── raw
│       ├── cardboard
│       ├── glass
│       ├── metal
│       ├── paper
│       ├── plastic
│       └── trash
├── evaluation
│   ├── metrics.json
│   └── plots
│       ├── confusion_matrix.png
│       ├── pred_preview.png
│       └── training_history.png
├── main.py
├── model
│   ├── history.npy
│   └── model.keras
├── notebook.ipynb
├── params.yaml
├── pyproject.toml
├── requirements-freeze.txt
├── requirements.txt
├── src
│   ├── evaluate.py
│   ├── prepare.py
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       └── seed.py
└── uv.lock

18 directories, 21 files
```

# DVC

Initialize DVC in the project with:
```bash
dvc init
```
You can activate the uv enviroment to use dvc commands.


