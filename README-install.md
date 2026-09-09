# 5ModIA — HDDL : installation

Environnement unique pour toute la série de TP, géré avec [uv](https://docs.astral.sh/uv/).

---

## INSTALLATION - Une seule fois au premier TP


### 1. Installer uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```


### 2. Récupérer le dépôt et installer

```bash
git clone https://plmlab.math.cnrs.fr/wikistat/5ModIA-HDDL.git
cd 5ModIA-HDDL
uv sync --all-groups
```

`uv` télécharge la bonne version de Python, crée `.venv/` et installe toutes les dépendances de la série. Compter quelques minutes et environ 4 Go la première fois
**C'est à faire une seule fois pour l'ensemble des TP.** 

> **NB:** Pour des questions de potentiels conflits, certains TPs pourront bénéficier d'un environnement dédié. Merci de suivre les consignes.


## 3. Vérifier que le GPU est vu

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Sur les machines des salles GMM 101 et 102, la seconde valeur doit être `True`. 
Sur un portable personnel sans GPU NVIDIA, elle sera `False` et les TP tourneront en CPU, plus lentement.


---

## USAGE - A chaque début de séance


### 1. Mettre à jour le dépot

```bash
git pull
```


### 2. Lancer

#### 2.a. Avec JupyterLab

```bash
uv run jupyter lab
```

Il n'y a **pas** d'environnement à activer : `uv run` s'en charge, et il remet l'environnement à jour avant de démarrer.

**Au début de chaque séance :** Fermer JupyterLab (si ce n'est déjà fait) puis le relancer par `uv run jupyter lab`. 
La mise à jour a lieu au lancement, pas pendant la session : un JupyterLab laissé ouvert d'une semaine sur l'autre tourne encore sur l'ancien environnement.

—> Après un `git pull`, il n'y a rien de plus à faire.


### 2.b Avec VS Code

VS Code lance le noyau directement, **sans passer par `uv`** : la mise à jour automatique décrite ci-dessus n'a donc pas lieu. 
Le `uv sync` devient obligatoire à chaque début de séance et après chaque `git pull`

```bash
uv sync  # indispensable : VS Code ne le fera pas
code .
```

Ensuite :
- ouvrir **la racine du dépôt** comme dossier de travail, sinon
  l'environnement n'est pas détecté ;
- installer les extensions *Python* et *Jupyter* ;
- si l'environnement n'est pas trouvé automatiquement :
  `Ctrl+Shift+P` -> *Python: Select Interpreter* -> `./.venv/bin/python` ;
- le sélecteur de noyau, en haut à droite du notebook, doit pointer sur ce
  même interpréteur.

**À refuser :** 
Si VS Code propose de créer un environnement virtuel, dire **non**.
Cela produirait un second `.venv` non géré, qui fonctionne un temps puis dérive.


### Trois règles

1. Ne jamais activer `.venv` à la main.
2. Ne jamais utiliser `pip` ni `uv pip install` : le paquet disparaîtra au prochain lancement, ou fera dériver l'environnement sans que ça se voie.
3. Pour un paquet supplémentaire, le demander plutôt que de l'installer soi-même : il sera ajouté proprement au dépôt.


### En cas de problème

À lancer **à la racine du dépôt**, là où se trouve `pyproject.toml` :

```bash
rm -rf .venv && uv sync --all-groups
```

L'environnement est reconstruit à l'identique en quelques minutes depuis le cache local. Rien n'est perdu : `.venv/` ne contient aucun travail, seulement des bibliothèques. 
C'est la première chose à essayer devant n'importe quel `ModuleNotFoundError` ou comportement inattendu.