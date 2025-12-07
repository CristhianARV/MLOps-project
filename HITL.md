## Rajouter une instance gpu au cluster

Il faut créer un nouveau cluster car pas gpu dans la zone par défaut.

```bash
export GCP_K8S_CLUSTER_NAME=mlops-trash-classification-cluster

export GCP_K8S_CLUSTER_ZONE=europe-west4-a

# Create the Kubernetes cluster
gcloud container clusters create \
    --machine-type=e2-standard-2 \
    --num-nodes=2 \
    --zone=$GCP_K8S_CLUSTER_ZONE \
    $GCP_K8S_CLUSTER_NAME

Note: Your Pod address range (`--cluster-ipv4-cidr`) can accommodate at most 1008 node(s).
Creating cluster mlops-trash-classification-cluster in europe-west4-a... Cluster is being health-checked (Kubernetes Control Plane is healthy)...done.                                                  
Created [https://container.googleapis.com/v1/projects/mlops-trash-classification/zones/europe-west4-a/clusters/mlops-trash-classification-cluster].
To inspect the contents of your cluster, go to: https://console.cloud.google.com/kubernetes/workload_/gcloud/europe-west4-a/mlops-trash-classification-cluster?project=mlops-trash-classification
kubeconfig entry generated for mlops-trash-classification-cluster.
NAME                                LOCATION        MASTER_VERSION      MASTER_IP     MACHINE_TYPE   NODE_VERSION        NUM_NODES  STATUS   STACK_TYPE
mlops-trash-classification-cluster  europe-west4-a  1.33.5-gke.1308000  34.34.75.206  e2-standard-2  1.33.5-gke.1308000  2          RUNNING  IPV4
```

```bash
kubectl get nodes
NAME                                                  STATUS   ROLES    AGE   VERSION
gke-mlops-trash-classifi-default-pool-1d9136ba-1sk9   Ready    <none>   177m   v1.33.5-gke.1308000
gke-mlops-trash-classifi-default-pool-1d9136ba-dfpd   Ready    <none>   177m   v1.33.5-gke.1308000
gke-mlops-trash-classificati-gpu-pool-64b3bf72-f068   Ready    <none>   104s   v1.33.5-gke.1308000

export K8S_NODE_1_NAME=gke-mlops-trash-classifi-default-pool-1d9136ba-1sk9
export K8S_NODE_2_NAME=gke-mlops-trash-classifi-default-pool-1d9136ba-dfpd
export K8S_NODE_3_NAME=gke-mlops-trash-classificati-gpu-pool-64b3bf72-f068

# Labelize the nodes
kubectl label nodes $K8S_NODE_1_NAME gpu=true
kubectl label nodes $K8S_NODE_2_NAME gpu=false
kubectl label nodes $K8S_NODE_3_NAME gpu=true
```

## Créer un pool de nœuds GPU

```bash
export GCP_K8S_GPU_NODE_POOL_NAME=gpu-pool
export GCP_K8S_GPU_NODE_COUNT=1
export GCP_K8S_GPU_MACHINE_TYPE=g2-standard-4
export GCP_K8S_GPU_ACCELERATOR_TYPE=nvidia-l4
export GCP_K8S_GPU_ACCELERATOR_COUNT=1

# Create the GPU node pool
gcloud container node-pools create $GCP_K8S_GPU_NODE_POOL_NAME \
    --cluster $GCP_K8S_CLUSTER_NAME \
    --zone $GCP_K8S_CLUSTER_ZONE \
    --machine-type $GCP_K8S_GPU_MACHINE_TYPE \
    --num-nodes $GCP_K8S_GPU_NODE_COUNT \
    --accelerator type=$GCP_K8S_GPU_ACCELERATOR_TYPE,count=$GCP_K8S_GPU_ACCELERATOR_COUNT \
    --node-labels=gpu=true \
    --scopes=https://www.googleapis.com/auth/cloud-platform

gcloud container node-pools create gpu-pool \
    --cluster=mlops-trash-classification-cluster \
    --zone=europe-west4-a \
    --machine-type=g2-standard-4 \
    --num-nodes=1 \
    --accelerator=type=nvidia-l4,count=1 \
    --scopes=https://www.googleapis.com/auth/cloud-platform










