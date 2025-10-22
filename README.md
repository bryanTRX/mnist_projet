# Projet MNIST – Reconnaissance de Chiffres Manuscrits
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)

![Interface de dessin MNIST](https://github.com/bryanTRX/mnist_projet/raw/main/results.png)

## Description

Ce projet propose une interface web interactive permettant de dessiner un chiffre (0-9) et d’obtenir sa prédiction en temps réel grâce à un modèle de réseau de neurones convolutif (CNN) entraîné sur le jeu de données MNIST.

L’application est construite avec :

- **Streamlit** : pour l’interface web interactive.
- **PyTorch** : pour le modèle CNN.
- **Streamlit Drawable Canvas** : pour dessiner les chiffres.

## Fonctionnalités

- Dessin de chiffres manuscrits sur un canvas interactif.
- Prédiction en temps réel du chiffre dessiné.
- Affichage de la distribution des probabilités pour chaque chiffre (0-9).
- Bouton pour réinitialiser le canvas.

## 🛠️ Prérequis

Avant de commencer, assurez-vous d’avoir installé Python 3.12 ou supérieur.

## Tester mon projet

Si vous voulez tester ce projet faites ceci :

### 1 - Clonage

```bash
git clone https://github.com/bryanTRX/mnist_projet.git
cd mnist_projet
```

### 2 - Installation des dépendances

```bash
pip install -r requirements.txt
```

L’application sera accessible à l’adresse suivante : http://localhost:8501

### 3 - Pour rouler la page web

```bash
python -m streamlit run app/draw_interface.py
```

Ce projet est sous licence MIT. Voir le fichier LICENSE 
