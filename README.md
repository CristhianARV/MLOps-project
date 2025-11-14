# ML Ops Project

Site de model evolution : https://cristhianarv.github.io/MLOps-project/

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

Compare changes in the pipeline with:
```bash
dvc params diff
```

And for metrics changes with:
```bash
dvc metrics diff
```
DVC displays the differences between HEAD and workspace, so you can easily compare the two iterations.


Generate a report with:
```bash
dvc plots diff --open
```
## Google Cloud Storage Backend for DVC

**Name du google project :** mlops-trash-classification

**Name du bucket :** mlops-cris-bucket 

Configure DVC to use a Google Storage remote bucket. The dvcstore is a user-defined path on the bucket. You can change it if needed:
```bash
dvc remote add -d data gs://mlops-cris-bucket/dvcstore
```

To get access to the GCS bucket
```bash
git clone <repo>

# Authenticate to GCP (their own Google account)
gcloud auth application-default login

# (optional) make ADC path explicit
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"

# Pull the data/artifacts
dvc pull
```

## Set up  access to S3 bucket of the cloud provider
Si des erreurs avec gcloud faire cela avant:
Mettre dans le fichier .venv/bin/activate
```bash
# === Google Cloud & DVC integration ===
export CLOUDSDK_CONFIG="$HOME/.config/gcloud"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
export BROWSER=/usr/bin/wslview
````


```bash
# Create the Google Service Account
gcloud iam service-accounts create google-service-account \
    --display-name="Google Service Account"

# Set the permissions for the Google Service Account
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:google-service-account@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Create the Google Service Account Key
gcloud iam service-accounts keys create ~/.config/gcloud/google-service-account-key.json \
    --iam-account=google-service-account@${GCP_PROJECT_ID}.iam.gserviceaccount.com
```

## Serve and deploy the model
### Serve model locally with BentoML
To serve the model locally using FastAPI, run the following command:
```bash
bentoml serve --working-dir ./src serve:TrashClassifierService
```

This command starts a local server that hosts the model, allowing you to make predictions via API calls. You can access the API at localhost on the default port 3000.

```json
{
  "prediction": "metal",
  "probabilities": {
    "cardboard": 0.01749773882329464,
    "glass": 0.17597563564777374,
    "metal": 0.6514637470245361,
    "paper": 0.01539438683539629,
    "plastic": 0.10146550089120865,
    "trash": 0.03820298984646797
  }
}
```
`Wrong prediction the model need more training and a change of architecture.`

### Build and push model with BentoML and Docker locally

BentoML model artifact is described in a bentofile.yaml file. Now that the bentofile.yaml is created, we can serve the model with commands like:
```bash
bentoml serve --working-dir ./src 
``` 

#### Build the BentoML model artifact
Before containerizing the model, we need to build the BentoML model artifact using the following command:
```bash
> bentoml build src

Successfully built Bento(tag="trash_classifier:znbg2n6arwfxaaav").

Next steps: 

* Deploy to BentoCloud:
    $ bentoml deploy trash_classifier:znbg2n6arwfxaaav -n ${DEPLOYMENT_NAME}

* Update an existing deployment on BentoCloud:
    $ bentoml deployment update --bento trash_classifier:znbg2n6arwfxaaav ${DEPLOYMENT_NAME}

* Containerize your Bento with `bentoml containerize`:
    $ bentoml containerize trash_classifier:znbg2n6arwfxaaav 

* Push to BentoCloud with `bentoml push`:
    $ bentoml push trash_classifier:znbg2n6arwfxaaav 
```
### Containerize the BentoML model artifact with Docker
To containerize the BentoML model artifact using Docker, run the following command:
```bash
> bentoml containerize trash_classifier:latest --image-tag trash-classifier:latest

Successfully built Bento container for "trash_classifier:latest" with tag(s) "trash-classifier:latest"
To run your newly built Bento container, run:
    docker run --rm -p 3000:3000 trash-classifier:latest
```

`:latest` is the tag of the BentoML model artifact. It is a symlink to the latest version of the BentoML model artifact.

### Test the containerized BentoML model artifact locally

The BentoML model artifact is now containerized. To verify its behavior, serve the model artifact locally by running the Docker image:
```bash
docker run --rm -p 3000:3000 trash-classifier:latest
```

### Create a container registry
#### Enable the Google Artifact Registry API
```bash
# Enable the Google Artifact Registry API
gcloud services enable artifactregistry.googleapis.com
```

#### Create the Google Container Registry
Export the repository name as an environment variable. Replace <my_repository_name> with a registy name of your choice. It has to be lowercase and words separated by hyphens.
```bash
export GCP_CONTAINER_REGISTRY_NAME=mlops-trash-classification-registry
```

Export the repository location as an environment variable. You can view the available locations at Cloud locations. You should ideally select a location close to where most of the expected traffic will come from. Replace <my_repository_location> with your own zone. For example, use europe-west6 for Switzerland (Zurich):

```bash
export GCP_CONTAINER_REGISTRY_LOCATION=europe-west6
```

Lastly, when creating the repository, remember to specify the repository format as `docker`.
```bash
# Create the Google Container Registry
gcloud artifacts repositories create $GCP_CONTAINER_REGISTRY_NAME \
    --repository-format=docker \
    --location=$GCP_CONTAINER_REGISTRY_LOCATION

Create request issued for: [mlops-trash-classification-registry]
Waiting for operation [projects/mlops-trash-classification/locations/europe-west6/operations/2eca15a6-e354-4826-acae-131b5f3c97d0] to complete...done.                                                  
Created repository [mlops-trash-classification-registry].
```
### Login to the remote Container Registry

Authenticate with the Google Container Registry

Configure gcloud to use the Google Container Registry as a Docker credential helper.
```bash
# Authenticate with the Google Container Registry
gcloud auth configure-docker ${GCP_CONTAINER_REGISTRY_LOCATION}-docker.pkg.dev
```

Ensure your GCP_PROJECT_ID variable is still correctly exported:
```bash
# Check the exported project ID
echo $GCP_PROJECT_ID
```
if empty, re-export it:
```bash
gcloud projects list
export GCP_PROJECT_ID=mlops-trash-classification
```

Export the container registry host:
```bash
export GCP_CONTAINER_REGISTRY_HOST=${GCP_CONTAINER_REGISTRY_LOCATION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_CONTAINER_REGISTRY_NAME}
```
### Publish the BentoML model artifact Docker image to the container registry

### Error de push denied !!!
```bash
nano ~/.docker/config.json
{
  "credsStore": "desktop.exe",
  "auths": {
    "europe-west6-docker.pkg.dev": {}
  }
}
```

```bash
gcloud auth print-access-token | docker login \
  -u oauth2accesstoken \
  --password-stdin \
  https://europe-west6-docker.pkg.dev
```


The BentoML model artifact Docker image can be published to the container registry with the following commands:
```bash
# Tag the local BentoML model artifact Docker image with the remote container registry host
docker tag trash-classifier:latest $GCP_CONTAINER_REGISTRY_HOST/trash-classifier:latest

# Push the BentoML model artifact Docker image to the container registry
docker push $GCP_CONTAINER_REGISTRY_HOST/trash-classifier:latest
```
The image is now available in the container registry. You can use it from anywhere using Docker or Kubernetes.

Open the container registry interface on the cloud provider and check that the artifact files have been uploaded.

## Build and publish the model with BentoML and Docker in the CI/CD pipeline
In this chapter, you will containerize and push the model to the container registry with the help of the CI/CD pipeline.


### Set up access to the container registry of the cloud provider

The container registry will need to be accessed inside the CI/CD pipeline to push the Docker image.

Update the Google Service Account and its associated Google Service Account Key to access Google Cloud from the CI/CD pipeline without your own credentials.

```bash
# Set the Cloud Storage permissions for the Google Service Account
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:google-service-account@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Set the Artifact Registry permissions for the Google Service Account
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:google-service-account@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.createOnPushWriter"
```

### Add contrainer resgistry CI/CD secrets

Add the container registry secret to access the container registry from the CI/CD pipeline. Depending on the CI/CD platform you are using, the process will be different:

Create the following new variables by going to the Settings section from the top header of your GitHub repository. Select Secrets and variables > Actions and select New repository secret:
```bash
echo $GCP_CONTAINER_REGISTRY_HOST
```

### Update the CI/CD pipeline configuration file

You will adjust the pipeline to build and push the the docker image to the container registry.



