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


Prepare stage for data preparation:
```bash
dvc stage add -n prepare \
    -p prepare \
    -d src/prepare.py -d src/utils/seed.py -d data/raw \
    -o data/prepared \
    uv run src/prepare.py data/raw data/prepared
```

The values of the parameters is prepare which includes all the prepare parameters referenced in the params.yaml file.

This stage has the src/prepare.py, the src/utils/seed.py and data/raw files as dependencies. If any of these files change, DVC will run the command python3.13 src/prepare.py data/raw data/prepared when using dvc repro.

The output of this command is stored in the data/prepared directory.

Train stage for model training:
```bash
dvc stage add -n train \
    -p train \
    -d src/train.py -d src/utils/seed.py -d data/prepared \
    -o model \
    uv run src/train.py data/prepared model
```

The values of the parameters is train which includes all the train parameters referenced in the params.yaml file.

This stage has the src/train.py, the src/utils/seed.py and data/prepared files as dependencies. If any of these files change, DVC will run the command uv run src/train.py data/prepared model when using dvc repro.

The output of this command is stored in the model directory.

Evaluate stage for model evaluation:
```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d model \
    --metrics evaluation/metrics.json \
    --plots evaluation/plots/confusion_matrix.png \
    --plots evaluation/plots/pred_preview.png \
    --plots evaluation/plots/training_history.png \
    uv run src/evaluate.py model data/prepared
```
This stage has the src/evaluate.py and the model files as dependencies. If any of these files change, DVC will run the command uv run src/evaluate.py model data/prepared when using dvc repro.

The script writes the model's metrics to evaluation/metrics.json, the confusion_matrix to evaluation/plots/confusion_matrix.png, the pred_preview to evaluation/plots/pred_preview.png and the training_history.png to evaluation/plots/training_history.png.

Visualize the pipeline with:
```bash
> dvc dag
+--------------+ 
| data/raw.dvc | 
+--------------+ 
        *        
        *        
        *        
  +---------+    
  | prepare |    
  +---------+    
        *        
        *        
        *        
    +-------+    
    | train |    
    +-------+    
        *        
        *        
        *        
  +----------+   
  | evaluate |   
  +----------+  
```

Important !!!

Execute the pipeline

Now that the pipeline is defined, you can execute it and reproduce the experiment with:
```bash
dvc repro
```





