# Projet ML: Machine Learning

Dans le cadre de notre formation Diginamic, notre 3ième projet est un projet de machine learning.

Ce projet de Machine Learning a été développé pour effectuer des tâches de prétraitement de données et de modélisation à l'aide de Streamlit et SQLAlchemy. Le projet est conçu pour travailler avec des données stockées dans une base de données PostgreSQL.

## Objectif du Projet

L'objectif principal de ce projet est de faciliter le processus de prétraitement des données et de modélisation en fournissant une interface utilisateur conviviale. Les principales fonctionnalités de ce projet sont les suivantes :

- Connexion à une base de données PostgreSQL.
- Sélection d'un jeu de données à partir de la base de données.
- Choix du type de modèle (Régression ou Classification).
- Prétraitement automatique des données, y compris la gestion des valeurs manquantes, des valeurs aberrantes, de l'encodage, de la standardisation, et de la division des données en ensembles d'entraînement et de test.
- Sélection du modèle d'apprentissage automatique et recherche des meilleurs hyperparamètres à l'aide de la validation croisée.
- Affichage des résultats de la modélisation, y compris les performances du modèle.

## Technologies Utilisées

Ce projet utilise plusieurs bibliothèques et technologies, notamment :

- Python
- Streamlit : pour la création de l'interface utilisateur.
- Pandas : pour la manipulation des données.
- NumPy : pour le support des tableaux et des calculs numériques.
- Plotly : pour la création de visualisations.
- SQLAlchemy : pour la connexion à la base de données PostgreSQL.
- psycopg2 : pour la gestion de la base de données PostgreSQL.
- Scikit-learn : pour l'apprentissage automatique et la modélisation.
- scipy : pour des fonctions statistiques avancées.

## Comment Utiliser le Projet
Pour exécuter ce projet sur votre machine, suivez ces étapes :

Ouvrez un terminal (invite de commande) sur votre ordinateur.

Créez un environnement virtuel (venv) en utilisant la commande suivante :
**python -m venv .venv**


Activez l'environnement virtuel en fonction de votre système d'exploitation :

Sur Windows :
**.venv\Scripts\activate**

Sur macOS et Linux :
**source .venv/bin/activate**

Installez les packages nécessaires en exécutant la commande suivante pour utiliser pip (le gestionnaire de paquets Python) :
**pip install -r requirements.txt**

Cela installera toutes les bibliothèques et dépendances nécessaires pour le projet.

Accédez au répertoire de votre projet en utilisant la commande cd :
**cd chemin/vers/Projet_ML-main**

Pour exécuter l'application Streamlit, utilisez la commande suivante :
**streamlit run main.py**

Remplacez main.py par le nom du fichier contenant votre application Streamlit (si vous modifiez le nom du fichier).

Une fois la commande lancée, une page "http://localhost:8501" s'ouvre. 
Vous pouvez maintenant explorer et interagir avec l'application streamlit. 

Fichier .streamlit/config.toml 
Ce fichier correspond à la configuration du thème/couleur de l'application streamlit. 
Il vous suffit d'avoir ce fichier sur votre machine dans le folder du projet, et streamlit prendra automatiquement en compte la configuration. 

## Auteur
Ce projet a été créé par Lucie GUILLAUD SAUMUR, Cécile CESA et Valentin PACHURKA.
