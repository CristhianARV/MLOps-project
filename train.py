import os, time, json, itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_datasets

def save_confusion_matrix(cm, class_names, out_png):
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
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(y.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return acc, mf1, np.array(y_true), np.array(y_pred)

def main():
    #  Config 
    epochs      = 10
    batch_size  = 32
    lr          = 3e-3
    weight_decay= 1e-4

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = torch.cuda.is_available()
    pin      = True if use_cuda else False
    workers  = 0  

    os.makedirs("reports", exist_ok=True)

    # Data 
    train_ds, val_ds, test_ds = get_datasets()
    if len(train_ds.classes) == 0:
        raise ValueError("⚠️ Aucune classe trouvée. Vérifie data/train/<classe>/*.jpg")
    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    # Modèle (ResNet50 gelé + tête)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    # Entraînement 
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs+1):
        start = time.time()
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", unit="batch", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/len(train_loader):.4f}")

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        dur = time.time() - start
        print(f"Epoch {epoch:02d} | loss={running/len(train_loader):.4f} | val_acc={val_acc:.3f} | val_macroF1={val_f1:.3f} | time={dur:.1f}s")

        history.append({"epoch": epoch, "train_loss": float(running/len(train_loader)),
                        "val_acc": float(val_acc), "val_macro_f1": float(val_f1), "time_s": dur})
        with open("reports/history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"state_dict": model.state_dict(),
                        "classes": class_names}, "best_baseline_resnet.pt")

    # Test + rapports
    ckpt = torch.load("best_baseline_resnet.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    test_acc, test_mf1, y_true, y_pred = evaluate(model, test_loader, device)
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    print(f"[TEST] acc={test_acc:.3f} | macroF1={test_mf1:.3f}")

    # Sauvegardes
    with open("reports/test_classification_report.txt", "w") as f:
        f.write(rep)
    np.savetxt("reports/test_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_matrix(cm, class_names, "reports/test_confusion_matrix.png")

    metrics = {
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_mf1),
        "classes": list(class_names),
        "best_val_macro_f1": float(best_f1)
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(" Rapports écrits dans ./reports :")
    print("  - history.json")
    print("  - test_classification_report.txt")
    print("  - test_confusion_matrix.csv / .png")
    print("  - metrics.json")
    print("  - best_baseline_resnet.pt")

if __name__ == "__main__":
    main()
