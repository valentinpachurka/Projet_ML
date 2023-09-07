import json

def get_model_advantages_disadvantages(json_filename, selected_algorithm):
    try:
        # Charger les données JSON depuis le fichier
        with open(json_filename, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        
        # Vérifier si l'algorithme sélectionné est présent dans les données JSON
        selected_algorithm_encoded = selected_algorithm.encode('utf-8').decode('utf-8')  # Correction de l'encodage
        if selected_algorithm_encoded in data:
            advantages = data[selected_algorithm_encoded].get("Avantages", [])
            disadvantages = data[selected_algorithm_encoded].get("Inconvénients", [])
            return advantages, disadvantages
        else:
            return [], []  # Retourner des listes vides si l'algorithme n'est pas trouvé
    except FileNotFoundError:
        print(f"Le fichier JSON '{json_filename}' n'a pas pu être chargé.")
        return [], []  # Retourner des listes vides en cas d'erreur de fichier

def get_metrics_info(json_filename, selected_metric):
    try:
        # Charger les données JSON depuis le fichier
        with open(json_filename, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        
        # Vérifier si l'algorithme sélectionné est présent dans les données JSON
        selected_metric_encoded = selected_metric.encode('utf-8').decode('utf-8')
        if selected_metric_encoded in data:
            messages = data[selected_metric_encoded]
            return messages
        else:
            message = "Pas d'éléments à retourner concernant cette métrique"
            return message
    except FileNotFoundError:
        message = f"Le fichier JSON '{json_filename}' n'a pas pu être chargé."
        return message

# Exemple d'utilisation de get_model_advantages_disadvantages
# json_filename = "info_model.json"
# selected_algorithm = "Régression Linéaire"

# advantages, disadvantages = get_model_advantages_disadvantages(json_filename, selected_algorithm)

# if advantages or disadvantages:
#     print(f"Avantages de '{selected_algorithm}':")
#     for advantage in advantages:
#         print("- " + advantage)

#     print(f"Inconvénients de '{selected_algorithm}':")
#     for disadvantage in disadvantages:
#         print("- " + disadvantage)
# else:
#     print(f"'{selected_algorithm}' non trouvé dans le fichier JSON.")

# Exemple d'utilisation de get_metrics_info
# json_filename = "info_metrics.json"
# selected_metric = "matrix_confusion"

# info_metric = get_metrics_info(json_filename, selected_metric)

# print(info_metric)