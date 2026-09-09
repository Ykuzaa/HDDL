#!/usr/bin/env bash
# Script d'installation d'un environnement PyTorch + DGL (CPU only) pour le TP GNN
# Usage :
#   bash pytorch-dgl-cpu.sh
#   ou :
#   bash pytorch-dgl-cpu.sh mon_env_perso
#
# Par défaut, l'environnement s'appellera "pytorch-dgl-cpu".

set -e

ENV_NAME="${1:-pytorch-dgl-cpu}"

echo "=============================================="
echo "  Création de l'environnement : $ENV_NAME"
echo "  Python 3.11 + PyTorch 2.4 (CPU) + DGL"
echo "=============================================="

# Vérifier que conda est disponible
if ! command -v conda >/dev/null 2>&1; then
    echo "La commande 'conda' n'est pas disponible."
    echo "Ouvre un terminal Anaconda / Miniconda ou ajoute conda au PATH."
    exit 1
fi

# Activer le hook conda pour pouvoir utiliser 'conda activate' dans un script
eval "$(conda shell.bash hook)"

# 1) Créer l'environnement conda
echo "➡️  Création de l'environnement conda (python=3.11)..."
conda create -n "$ENV_NAME" python=3.11 -y

# 2) Activer l'environnement
echo "➡️  Activation de l'environnement '$ENV_NAME'..."
conda activate "$ENV_NAME"

# 3) Mettre pip à jour
echo "➡️  Mise à jour de pip..."
pip install --upgrade pip

# 4) Installer PyTorch (CPU only)
echo "➡️  Installation de PyTorch 2.4.0 (CPU)..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cpu

# 5) Installer DGL (CPU) compatible torch 2.4.*
echo "➡️  Installation de DGL (CPU) pour torch 2.4..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html

# 6) Installer Jupyter + packages utiles pour le TP
echo "➡️  Installation de ipykernel, jupyterlab et bibliothèques scientifiques..."
pip install ipykernel jupyterlab matplotlib seaborn scikit-learn pandas

# 7) Enregistrer le kernel pour Jupyter
echo "➡️  Enregistrement du kernel Jupyter..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python (PyTorch+DGL CPU)"

echo "Installation terminée."
echo
echo "Vous pouvez maintenant :"
echo "  - Lancer Jupyter (jupyter lab ou jupyter notebook)"
echo "  - Choisir le kernel : 'Python (PyTorch+DGL CPU)'"
echo
echo "Pour activer l'environnement en terminal :"
echo "  conda activate $ENV_NAME"
