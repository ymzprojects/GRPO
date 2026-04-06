# Optimisation de Politique par Groupe (GRPO) : Résolution du Jeu de Taquin

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Reinforcement Learning](https://img.shields.io/badge/RL-GRPO-green)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)

## 📌 Présentation du Projet

Ce dépôt explore l'algorithme **Group Relative Policy Optimization (GRPO)** appliqué à la résolution du jeu de taquin (3x3). L'objectif est de valider l'efficacité de cette méthode d'apprentissage par renforcement, initialement conçue pour l'alignement des modèles de langage, sur un problème de logique combinatoire.

Le projet compare deux stratégies d'apprentissage distinctes :
1.  **Exploration Autonome :** Le modèle apprend sans aucune aide extérieure, uniquement par essai-erreur.
2.  **Guidage Expert :** Introduction de trajectoires expertes générées par inversion du mélange pour accélérer la convergence.

---

## 🚀 Fonctionnalités Clés

### 1. Implémentation de l'Algorithme GRPO
Une approche moderne qui optimise la politique sans nécessiter de réseau critique séparé, simplifiant ainsi l'architecture globale.

* **Calcul de l'Avantage Relatif :** Estimation de l'avantage par normalisation des récompenses au sein d'un groupe de trajectoires issues d'un même état initial.
* **Politique Dense & ResNet :** Comparaison entre une architecture de perceptron multicouche et une structure résiduelle pour capturer les relations spatiales de la grille.
* **Buffer Multi-Files :** Gestion optimisée de la mémoire pour l'entraînement par lots (batch) sur les transitions collectées.

### 2. Environnement de Simulation (Taquin)
Modélisation complète des mécaniques du puzzle 3x3.

* **Récompense Enrichie :** Utilisation de la distance de Manhattan combinée à un bonus de résolution pour guider l'agent.
* **Générateur de Défis :** Création d'états de départ par mélanges aléatoires réversibles, permettant d'extraire des trajectoires de référence.
* **Mise à Jour Dynamique :** Adaptation automatique du nombre d'échantillons et du taux d'exploration durant l'entraînement.

---

## 📂 Structure du Dépôt

| Fichier | Description |
| :--- | :--- |
| **`grpo_taquin_resolution.ipynb`** | **Notebook Principal.** Contient les deux phases d'entraînement (Exploration vs Guidage), les visualisations et l'évaluation finale. |
| **`grpo.py`** | **Script Modulaire.** Regroupe les classes de modèles, le tampon de données (buffer) et la fonction de coût GRPO. |

---

## 📊 Analyse des Performances

### Résultats Comparatifs
L'évaluation sur 100 états de test montre une progression notable grâce au guidage :
* **Taux de résolution (Exploration) :** Environ **64%**. Le modèle parvient à comprendre les règles mais peine à résoudre les configurations les plus complexes.
* **Taux de résolution (Guidé) :** Environ **86%**. L'imitation initiale des trajectoires expertes stabilise la politique et augmente significativement le succès.

### Instabilité et Phénomène d'Effondrement
Les courbes d'apprentissage révèlent une instabilité inhérente aux méthodes de politique. Nous observons fréquemment un **effondrement de l'apprentissage** (catastrophic forgetting) : après avoir atteint un pic de performance, le modèle peut brutalement "oublier" ses acquis et voir ses récompenses s'effondrer. Ce comportement souligne la sensibilité des hyperparamètres dans l'algorithme GRPO.

---

## 🛠️ Pistes d'Exploration & Améliorations

Pour stabiliser l'apprentissage et dépasser le seuil des 90%, plusieurs axes sont envisagés :
* **Contrainte KL Stricte :** Intégrer une divergence de Kullback-Leibler plus rigide pour empêcher des mises à jour de politique trop brutales.
* **Relecture d'Expérience Sélective (PER) :** Prioriser les trajectoires réussies dans le buffer pour maintenir les acquis.
* **Planification de l'Apprentissage :** Utiliser un `ReduceLROnPlateau` et du Gradient Clipping pour éviter les dérives de poids lors des crashs de performance.
* **Architecture Transformer :** Explorer l'attention spatiale pour mieux modéliser les dépendances entre les cases.

---

## 🚀 Installation & Utilisation

### Prérequis
* Google Colab ou un environnement Python 3.9+ local avec GPU.
* Bibliothèques : `torch`, `numpy`, `matplotlib`, `tqdm`.

### Exécution
1. Ouvrez le notebook `grpo_project.ipynb`.
2. Exécutez les cellules d'initialisation pour charger les fonctions utilitaires.
3. Lancez les phases d'entraînement 1 et 2 pour observer les différences de convergence.
