# dashboard.py

from pathlib import Path
import json
import subprocess

import streamlit as st
import pandas as pd
import yaml

def load_metrics():
    metrics_path = Path("evaluation/metrics.json")
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def load_params():
    params_path = Path("params.yaml")
    if not params_path.exists():
        return {}
    return yaml.safe_load(params_path.read_text())


def save_params(params: dict):
    params_path = Path("params.yaml")
    params_path.write_text(yaml.safe_dump(params, sort_keys=False))


def count_hitl_queue():
    to_label_dir = Path("data/hitl/to_label")
    if not to_label_dir.exists():
        return 0, []
    images = [p for p in to_label_dir.iterdir() if p.is_file()]
    return len(images), images


def count_hitl_labeled():
    labeled_root = Path("data/hitl/labeled")
    if not labeled_root.exists():
        return {}
    stats = {}
    for class_dir in labeled_root.iterdir():
        if class_dir.is_dir():
            count = len([p for p in class_dir.iterdir() if p.is_file()])
            stats[class_dir.name] = count
    return stats


def show_evaluation_plots():
    plots_root = Path("evaluation/plots")
    if not plots_root.exists():
        st.info("Aucun plot trouvé dans evaluation/plots/. Lancer d'abord `dvc repro evaluate`.")
        return

    training_history = plots_root / "training_history.png"
    pred_preview = plots_root / "pred_preview.png"
    confusion_matrix = plots_root / "confusion_matrix.png"

    if training_history.exists():
        st.subheader("Courbe de loss (entraînement / validation)")
        st.image(str(training_history))

    if confusion_matrix.exists():
        st.subheader("Matrice de confusion")
        st.image(str(confusion_matrix))

    if pred_preview.exists():
        st.subheader("Aperçu des prédictions sur le set de test")
        st.image(str(pred_preview))


def run_hitl_cycle_if_available():
    script = Path("run_hitl_cycle.py")
    if not script.exists():
        st.warning("Le script `run_hitl_cycle.py` n'existe pas ")
        return

    if st.button("Lancer un cycle HITL complet"):
        with st.spinner("Cycle HITL en cours..."):
            try:
                subprocess.run(["python3.12", "run_hitl_cycle.py"], check=True)
                st.success("Cycle HITL terminé. Recharger la page pour voir les nouvelles métriques.")
            except subprocess.CalledProcessError as e:
                st.error(f"Erreur lors de l'exécution du cycle HITL : {e}")


# ---------- UI ----------

st.set_page_config(
    page_title="Waste Classifier – HITL Dashboard",
    layout="wide",
)

st.title("🗑️ Waste Classifier – Dashboard MLOps & Human-in-the-Loop")


# organisation de l'interface
tab_overview, tab_hitl, tab_settings = st.tabs(
    ["Modèle & performance", "HITL & données", "Paramètres / Orchestration"]
)

# : Modèle & performance 
with tab_overview:
    st.header("Performance du modèle")

    metrics = load_metrics()
    if metrics is None:
        st.info("Aucune métrique trouvée. Lance `dvc repro evaluate` pour générer evaluation/metrics.json.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Validation loss", f"{metrics.get('val_loss', 0):.4f}")
        with col2:
            st.metric("Validation accuracy", f"{metrics.get('val_acc', 0)*100:.2f} %")

    st.markdown("---")
    show_evaluation_plots()

# HITL & données 
with tab_hitl:
    st.header("Suivi de la boucle Human-in-the-Loop")

    # File d'attente HITL
    st.subheader("File d'attente HITL (images à relabelliser)")
    n_to_label, images_to_label = count_hitl_queue()
    st.write(f"Images en attente de re-labellisation dans `data/hitl/to_label/` : **{n_to_label}**")

    if n_to_label > 0:
        st.caption("Aperçu des premières images à relabelliser :")
        cols = st.columns(4)
        for i, img_path in enumerate(images_to_label[:8]):
            with cols[i % 4]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)

    st.markdown("---")

    # Images déjà relabellisées
    st.subheader("Images relabellisées et réinjectées dans le dataset")
    labeled_stats = count_hitl_labeled()
    if not labeled_stats:
        st.write("Aucune image HITL relabellisée trouvée dans `data/hitl/labeled/`.")
    else:
        df = pd.DataFrame(
            [{"Classe": c, "Nombre d'images": n} for c, n in labeled_stats.items()]
        )
        st.table(df)

        st.bar_chart(df.set_index("Classe"))

# Paramètres & orchestration 
with tab_settings:
    st.header("Paramètres HITL & orchestration")

    params = load_params()
    if not params:
        st.error("Impossible de lire params.yaml")
    else:
        hitl_cfg = params.get("hitl", {})
        current_threshold = float(hitl_cfg.get("threshold", 0.75))

        st.subheader("Seuil de confiance pour envoyer une image en HITL")
        st.write(
            "Les images dont la probabilité max est **inférieure** à ce seuil "
            "sont envoyées dans la file `data/hitl/to_label/`."
        )

        new_threshold = st.slider(
            "Seuil HITL",
            min_value=0.0,
            max_value=1.0,
            value=current_threshold,
            step=0.01,
        )

        col_save, col_info = st.columns([1, 3])
        with col_save:
            if st.button("Enregistrer le nouveau seuil dans params.yaml"):
                params.setdefault("hitl", {})
                params["hitl"]["threshold"] = float(new_threshold)
                save_params(params)
                st.success(f"Nouveau seuil HITL enregistré : {new_threshold:.2f}")

        with col_info:
            st.markdown(
                """
                - Un seuil **plus bas** → moins d'images partent en HITL, mais plus sûres.
                - Un seuil **plus haut** → plus d'images partent en HITL (plus de travail humain, mais meilleure couverture des cas difficiles).
                """
            )

    st.markdown("---")
    st.subheader("Orchestration d'un cycle HITL complet")
    run_hitl_cycle_if_available()
