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


## Prétraitement des Données

Ce projet de Machine Learning inclut des fonctionnalités de prétraitement des données pour garantir que les données sont prêtes à être utilisées dans la modélisation. Voici comment le prétraitement des données est effectué :

1. **Connexion à la Base de Données** : L'application se connecte à une base de données PostgreSQL en utilisant les informations d'identification spécifiées. Cela permet de récupérer les données pour l'analyse.

2. **Sélection du Jeu de Données** : L'utilisateur peut choisir un jeu de données spécifique à partir de la base de données, en fonction des options disponibles.

3. **Gestion des Valeurs Manquantes** : Les valeurs manquantes dans le jeu de données sont gérées de manière automatique pour assurer la qualité des données. 

4. **Détection et Gestion des Valeurs Aberrantes** : Les valeurs aberrantes sont identifiées et gérées pour éviter qu'elles n'affectent négativement la modélisation.

5. **Encodage des Données** : Les données catégorielles sont encodées pour les préparer à l'entrée dans les modèles d'apprentissage automatique.

6. **Standardisation des Données** : Les données sont standardisées pour garantir que toutes les fonctionnalités sont à la même échelle.

7. **Division des Données** : Les données sont divisées en ensembles d'entraînement et de test pour l'apprentissage automatique.

## Modélisation

La modélisation est l'une des étapes clés de ce projet de Machine Learning, et elle permet de construire et d'évaluer des modèles d'apprentissage automatique. Voici comment la modélisation est effectuée :

1. **Choix du Type de Modèle** : L'utilisateur peut choisir entre les types de modèles de régression ou de classification en fonction de la nature du problème.

2. **Sélection du Modèle** : En fonction du type de modèle choisi, l'utilisateur peut sélectionner un algorithme spécifique parmi les options disponibles. Des informations sur les avantages et les inconvénients du modèle choisi sont fournies pour aider à la décision.

3. **Recherche des Meilleurs Hyperparamètres** : L'application propose une option de recherche automatique des meilleurs hyperparamètres en utilisant la validation croisée. L'utilisateur peut également choisir de spécifier manuellement les hyperparamètres s'il le souhaite.

4. **Évaluation du Modèle** : Une fois que le modèle est formé, plusieurs métriques de performance sont calculées, notamment le coefficient de détermination (R²), l'erreur quadratique moyenne (RMSE), l'erreur absolue moyenne (MAE), la matrice de confusion (pour la classification), la courbe ROC-AUC (pour la classification), etc.

5. **Visualisation des Résultats** : Les résultats de la modélisation, y compris les métriques de performance et les graphiques pertinents, sont affichés à l'utilisateur pour évaluation.

En suivant ces étapes de prétraitement des données et de modélisation, ce projet vise à fournir une interface utilisateur conviviale pour simplifier le processus de Machine Learning et aider les utilisateurs à obtenir des informations exploitables à partir de leurs données.

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

Fichier .streamlit/config.toml

Ce fichier correspond à la configuration du thème/couleur de l'application streamlit. 
Il vous suffit d'avoir ce fichier sur votre machine dans le folder du projet, et streamlit prendra automatiquement en compte la configuration. 

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

Vous pouvez retrouver sous le lien "" hébergé sous streamlit l'application disponible au public.


## Auteur
Ce projet a été créé par Lucie GUILLAUD SAUMUR, Cécile CESA et Valentin PACHURKA.
