import os, json, itertools
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from dataset import get_datasets

def save_confusion_matrix(cm, class_names, out_png):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("reports", exist_ok=True)

    # Datasets & loader
    _, _, test_ds = get_datasets()
    class_names = test_ds.classes
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # Modèle
    ckpt = torch.load("best_baseline_resnet.pt", map_location=device)
    if "classes" in ckpt:
        class_names = ckpt["classes"]  

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    # Évaluation
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred.extend(model(x).argmax(1).cpu().tolist())
            y_true.extend(y.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    # Sauvegardes
    with open("reports/test_classification_report.txt","w") as f: f.write(rep)
    np.savetxt("reports/test_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_matrix(cm, class_names, "reports/test_confusion_matrix.png")
    metrics = {"test_accuracy": float(acc), "test_macro_f1": float(mf1), "classes": list(class_names)}
  
    try:
        hist = json.load(open("reports/history.json"))
        best_val = max(h.get("val_macro_f1", 0.0) for h in hist)
        metrics["best_val_macro_f1"] = float(best_val)
    except Exception:
        pass
    with open("reports/metrics.json","w") as f: json.dump(metrics, f, indent=2)

    print(" Artefacts générés dans ./reports :")
    print("  - test_classification_report.txt")
    print("  - test_confusion_matrix.csv / .png")
    print("  - metrics.json")

if __name__ == "__main__":
    main()
