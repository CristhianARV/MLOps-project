import subprocess
from pathlib import Path

EXPORT_PATH = Path("data/hitl/exports/labelstudio_export.json")
RAW_ROOT = Path("data/raw")
HITL_LABELED_ROOT = Path("data/hitl/labeled")


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    # 1) Générer les prédictions et sélectionner les images peu confiantes
    print("=== Step 1: DVC -> predict_unlabeled + select_hitl ===")
    run(["dvc", "repro", "select_hitl"])

    to_label_dir = Path("data/hitl/to_label")
    n_to_label = len(list(to_label_dir.glob("*")))
    print(f"\n Images à annoter (HITL) dans {to_label_dir}: {n_to_label}")

    if n_to_label == 0:
        print("Rien à annoter pour l’instant, fin du cycle HITL.")
        return

    # 2) Instructions pour Label Studio
    print(
        f"""
=== Step 2: Annoter dans Label Studio ===

1. Lance Label Studio (si ce n'est pas déjà fait), par ex.:
   label-studio start

2. Crée ou ouvre un projet pour le waste classifier.

3. Importe les images depuis:
   {to_label_dir.resolve()}

4. Annote/corrige les labels (cardboard, glass, metal, paper, plastic, trash).

5. Exporte les annotations au format JSON dans:
   {EXPORT_PATH.resolve()}

Quand c'est fait, appuie sur Entrée pour continuer.
"""
    )
    input("Appuie sur Entrée quand l'export JSON est prêt... ")

    if not EXPORT_PATH.exists():
        print(f"Fichier d'export introuvable : {EXPORT_PATH}")
        print("Vérifie le chemin d'export dans Label Studio.")
        return

    # 3) Merge des labels HITL dans data/raw et data/hitl/labeled
    print("=== Step 3: Merge des labels HITL dans le dataset ===")
    run(
        [
            "python3.12",
            "src/merge_hitl_labels.py",
            str(EXPORT_PATH),
            str(RAW_ROOT),
            str(HITL_LABELED_ROOT),
        ]
    )

    # 4) Retrain + evaluate avec DVC
    print("=== Step 4: Retrain + evaluate via DVC ===")
    run(["dvc", "repro", "evaluate"])

    print(
        "\nCycle HITL terminé : modèle ré-entraîné et évalué."
        "\n   Tu peux vérifier evaluation/metrics.json et les plots dans evaluation/plots/."
    )


if __name__ == "__main__":
    main()
