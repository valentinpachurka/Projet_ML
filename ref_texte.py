class Dictext:
    @staticmethod
    def print_algorithm_advantages_disadvantages(selected_algorithm):
        advantages_disadvantages = {
            "Régression Linéaire": {
                "Avantages": ["Facile à comprendre et à implémenter", "Modélise la relation linéaire entre les variables d'entrée et la cible."],
                "Inconvénients": ["Ne gère pas bien les relations complexes", "Nécessite un réglage de l'hyperparamètre alpha (régularisation)."]
            },
            "Régression Ridge": {
                "Avantages": ["Réduit les effets des outliers et améliore la généralisation", "Modélise de manière stable en réduisant l'overfitting."],
                "Inconvénients": ["Ne gère pas bien la sélection des variables", "Nécessite un réglage de l'hyperparamètre alpha (régularisation)."]
            },
            "Régression Lasso": {
                "Avantages": ["Sélection automatique des variables", "Élimine des features."],
                "Inconvénients": ["Instable avec des corrélations élevées et sensible aux outliers", "Nécessite un réglage de l'hyperparamètre alpha (régularisation)."]
            },
            "Régression ElasticNet": {
                "Avantages": ["Gère les problèmes de corrélation et de multicolinéarité", "Contrôle et flexibilité entre Lasso et Ridge."],
                "Inconvénients": ["Complexité accrue et choix des valeurs d'hyperparamètres importants", "Nécessite un réglage de l'alpha (régularisation) et du L1_ratio."]
            },
            "Régression Logistique": {
                "Avantages": ["Bon pour la classification binaire", "Modélise la probabilité d'appartenance à une classe en utilisant la fonction logistique."],
                "Inconvénients": ["Ne fonctionne pas bien avec des données non linéaires", "Nécessite un réglage du taux d'apprentissage et de la régularisation."]
            },
            "Arbres de Décision": {
                "Avantages": ["Interprétables, gère bien les non-linéarités", "Possibilité de limiter la profondeur de l'arbre."],
                "Inconvénients": ["Tendance à overfitter avec des arbres profonds", "Sensible aux données bruitées."]
            },
            "Forêts aléatoires": {
                "Avantages": ["Réduit le surapprentissage, performances élevées", "Agrège les prédictions de plusieurs arbres de décision."],
                "Inconvénients": ["Moins interprétables que les arbres simples", "Nécessite de régler le nombre d'arbres, la profondeur et le critère de division."]
            },
            "SVM (Support Vector Machines)": {
                "Avantages": ["Bon pour les données non linéaires, efficace avec peu de données", "Trouve l'hyperplan qui maximise la marge entre les classes."],
                "Inconvénients": ["Sensible au choix du noyau et des hyperparamètres", "Nécessite de régler le paramètre de régularisation C et le type de noyau."]
            },
            "k-Means": {
                "Avantages": ["Simple et rapide, efficace pour la clustering", "Regroupe les données en clusters en minimisant la distance entre les points et le centre de leur cluster."],
                "Inconvénients": ["Sensible au nombre de clusters initial", "Nécessite de régler le nombre de clusters et le critère d'affectation."]
            },
            "Réseau de Neurones": {
                "Avantages": ["Performances élevées pour des tâches complexes", "Modèle basé sur le cerveau humain, composé de couches de neurones interconnectés."],
                "Inconvénients": ["Besoin d'une grande quantité de données", "Nécessite de régler l'architecture (nombre de couches, neurones, fonctions d'activation)."]
            },
            "Naïve Bayes": {
                "Avantages": ["Efficace avec peu de données, interprétable", "Utilise le théorème de Bayes pour prédire la probabilité d'appartenance à une classe."],
                "Inconvénients": ["Supposition d'indépendance des caractéristiques", "Peut nécessiter un lissage optionnel."]
            },
            "Gradient Boosting": {
                "Avantages": ["Performances élevées, gère bien les données bruitées", "Construit un modèle en agrégeant séquentiellement des modèles plus simples."],
                "Inconvénients": ["Sensible à l'overfitting, temps d'entraînement élevé", "Nécessite de régler le taux d'apprentissage, la profondeur et le nombre d'estimateurs."]
            },
            "PCA (Principal Component Analysis)": {
                "Avantages": ["Réduit la dimensionnalité, facilite la visualisation", "Transforme les données en un nouvel espace en identifiant les axes principaux de variabilité."],
                "Inconvénients": ["Perte d'interprétabilité des caractéristiques", "Nécessite de régler le nombre de composantes principales."]
            },
            "LDA (Linear Discriminant Analysis)": {
                "Avantages": ["Bon pour la classification multiclasse", "Réduit la dimension tout en maximisant la séparabilité entre les classes."],
                "Inconvénients": ["Sensible à l'échelle des données", "Nécessite de régler le nombre de composantes discriminantes."]
            },
            "Réseaux de Neurones Convolutifs (CNN)": {
                "Avantages": ["Performances élevées pour la vision par ordinateur", "Conçus pour traiter des données structurées en grille telles que les images."],
                "Inconvénients": ["Besoin de beaucoup de données, temps d'entraînement élevé", "Nécessite de régler l'architecture (couches convolutives, de pooling, etc.)."]
            },
            "XGBoost": {
                "Avantages": ["Hautes performances, régularisation intégrée", "Version optimisée du gradient boosting avec des améliorations spécifiques."],
                "Inconvénients": ["Nécessite le réglage des hyperparamètres", "Nécessite de régler le taux d'apprentissage, la profondeur et le nombre d'estimateurs."]
            }
        }

        if selected_algorithm in advantages_disadvantages:
            advantages = "\n".join(advantages_disadvantages[selected_algorithm]["Avantages"])
            disadvantages = "\n".join(advantages_disadvantages[selected_algorithm]["Inconvénients"])
        return advantages, disadvantages

    @staticmethod
    def print_metrics_info(selected_metrics):
        metrics_info = {
            "matrix_confusion": "La matrice de confusion est un outil essentiel pour évaluer les performances d'un modèle de classification. Elle permet de mesurer la capacité du modèle à distinguer les classes positives et négatives. Les éléments de la matrice sont :\n"
                                "- Vrais Positifs (VP) : Prédictions correctes de la classe positive.\n"
                                "- Vrais Négatifs (VN) : Prédictions correctes de la classe négative.\n"
                                "- Faux Positifs (FP) : Prédictions erronées de la classe positive (erreur de type I).\n"
                                "- Faux Négatifs (FN) : Prédictions erronées de la classe négative (erreur de type II).\n",
            "report_classif": "Le rapport de classification fournit des métriques de performance détaillées pour le modèle de classification. Chaque métrique a une signification spécifique :\n"
                                "- La précision mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives.\n"
                                "- Le rappel mesure la proportion de vrais positifs correctement identifiés parmi toutes les observations réellement positives.\n"
                                "- Le F1-score est une moyenne harmonique de la précision et du rappel, utile pour trouver un équilibre entre les deux.\n"
                                "- Le support indique le nombre d'occurrences de chaque classe dans l'ensemble de test, aidant à comprendre la distribution des classes.\n"
                                "Le rapport de classification est essentiel pour évaluer la performance d'un modèle de classification, en particulier dans les tâches multiclasse.\n",
            "roc_auc": "La courbe ROC (Receiver Operating Characteristic) évalue la capacité du modèle à distinguer les classes positives et négatives, couramment utilisée en classification binaire.\n"
                                "- La courbe ROC trace la sensibilité (taux de vrais positifs) par rapport à la spécificité (1 - taux de faux positifs) pour différents seuils de classification.\n"
                                "- Une courbe ROC qui se rapproche du coin supérieur gauche indique une meilleure performance du modèle pour discriminer les classes.\n"
                                "- L'AUC (Area Under the Curve) de la courbe ROC mesure la performance globale du modèle, avec une AUC de 0,5 indiquant une performance aléatoire et 1,0 une performance parfaite.\n"
                                "- La courbe ROC permet d'ajuster les seuils de classification pour trouver le meilleur compromis entre sensibilité et spécificité.\n"
                                "- En résumé, une courbe ROC élevée et une grande AUC indiquent une meilleure capacité du modèle à faire des prédictions précises.\n",
            "reg_metrics": "Les métriques de régression fournissent des informations sur la performance d'un modèle de régression. Voici les métriques les plus couramment utilisées :\n"
                                "- RMSE (Root Mean Squared Error) : Mesure la racine carrée de la moyenne des carrés des erreurs, indiquant l'écart moyen entre les valeurs prédites et les valeurs réelles.\n"
                                "- R² (Coefficient de détermination) : Mesure la proportion de la variance totale de la variable dépendante expliquée par le modèle. Une valeur de 1 indique un ajustement parfait.\n"
                                "- MAE (Mean Absolute Error) : Mesure la moyenne des valeurs absolues des erreurs, indiquant l'écart moyen entre les valeurs prédites et les valeurs réelles.\n",
            "regression_plot": "Le diagramme de dispersion est un outil visuel qui montre la relation entre les valeurs réelles et les valeurs prédites d'un modèle de régression. Il est utile pour évaluer la qualité des prédictions.\n",
            "regression_coefficient_hist": "L'histogramme des coefficients de régression est un graphique qui montre la distribution des coefficients attribués à chaque variable explicative dans un modèle de régression linéaire. Il permet de comprendre l'importance relative de chaque variable dans la prédiction.\n",
            "courbe_apprentissage": "La "courbe d'apprentissage" est un graphique montrant l'évolution de la performance d'un modèle en fonction de la taille de l'ensemble d'entraînement. Elle permet de détecter le sous-apprentissage et le sur-apprentissage, aidant ainsi à ajuster la complexité du modèle pour de meilleures performances."
    }

        metrics_info_text = "\n\n".join(metrics_info[metric] for metric in selected_metrics)
        return metrics_info_text
