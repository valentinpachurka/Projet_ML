import streamlit as st
import psycopg2
import sqlalchemy
import pandas as pd
from preprocessing import DataPreprocessor
from modeles import Modelisation, DictModels
from info_text import get_model_advantages_disadvantages, get_metrics_info


class MachineLearningApp:
    def __init__(self):
        self.table_list, self.engine = self.connection_db()
        self.df_process = None
        self.model = None
        self.selected_model = None

    def connection_db(self):
        host = "ec2-34-247-16-250.eu-west-1.compute.amazonaws.com"
        database = "d1fqoktf0gl90p"
        user = "xpfxvuvcndvbve"
        port = 5432
        password = "43b5e0de771549a5cb3117f84603628575b85328a0aecd350b017dcbf4534ddb"
        URI = "postgresql://xpfxvuvcndvbve:43b5e0de771549a5cb3117f84603628575b85328a0aecd350b017dcbf4534ddb@ec2-34-247-16-250.eu-west-1.compute.amazonaws.com:5432/d1fqoktf0gl90p"
        engine = sqlalchemy.create_engine(URI)
        table_list = []

        try:
            connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )

            cursor = connection.cursor()
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            cursor.execute(query)
            tables = cursor.fetchall()

            for table in tables:
                table_list.append(table[0])

            cursor.close()
            connection.close()

        except Exception as e:
            st.error("Erreur lors de la connexion à la base de données: " + str(e))

        return table_list, engine

    def run(self):
        self.intro()
        self.header()
        self.sidebar()

    def intro(self):
        st.set_page_config(
            page_title="Projet Machine Learning",
            layout="wide",
            initial_sidebar_state="auto"
        )

    def header(self):
        st.title('Machine Learning')

    def sidebar(self):
        try:
            selected_table = st.sidebar.selectbox("Choix d'un dataset", self.table_list)
            model_type = st.sidebar.selectbox("Choix du type de modèle", ["Régression", "Classification"])

            if model_type == "Régression":
                self.selected_model = st.sidebar.selectbox("Choix du modèle",
                                                      list(DictModels.get_regressors_dict()["algorithme"].keys()))
            elif model_type == "Classification":
                self.selected_model = st.sidebar.selectbox("Choix du modèle",
                                                      list(DictModels.get_classifiers_dict()["algorithme"].keys()))
            try:
                json_filename = "info_model.json"
                advantages, disadvantages = get_model_advantages_disadvantages(json_filename, self.selected_model)
                st.sidebar.info("".join(advantages))
                st.sidebar.info("".join(disadvantages))
            except Exception:
                st.sidebar.error(
                    f"Pour le moment, nous ne pouvons fournir \nles avantages et inconvenients du modèle {self.selected_model}")

            split_ratio = st.sidebar.slider("Ratio de division (test size)", 0.1, 0.5, 0.2, step=0.1)

            self.preprocess_data(selected_table, model_type, split_ratio)

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

    def preprocess_data(self, selected_table, model_type, split_ratio):
        try:
            try:
                df_process = DataPreprocessor(selected_table, self.engine, split_ratio)
                columns = df_process.load_data().columns
                expander = st.sidebar.expander("Choix des colonnes")
                selected_columns = expander.multiselect(" ", list(columns), default=list(columns))
                df_process.preprocess(selected_columns)
            except Exception as e:
                st.error(f"Erreur lors du preprocessing: " + str(e))

            if model_type == "Classification":
                n_class = df_process.original_dataframe["target"].nunique()
                st.subheader(f"Nombre de classes distinctes : {n_class}")
                liste = df_process.original_dataframe["target"].unique()
                liste_str = "Les classes : " + (" ".join(str(value) for value in sorted(liste)))
                st.caption(liste_str)

            st.subheader("Aperçu des données :", selected_table)
            st.write(df_process.original_dataframe.shape)
            st.write(df_process.original_dataframe.head(9))
            st.subheader("Statistiques descriptives :")
            st.write(df_process.original_dataframe.describe())
            st.subheader("Données prétraitées: ")
            st.write(df_process.dataframe.shape)
            st.write(df_process.dataframe.head(9))

            st.title("Prétraitement de Données")
            X_train, X_test, y_train, y_test = df_process.X_train, df_process.X_test, df_process.y_train, df_process.y_test,
            if X_train is not None:
                st.subheader("Ensembles de données résultants:")
                st.write(f"X_train : {X_train.shape}")
                st.write(f"X_test : {X_test.shape}")
                st.write(f"y_train : {y_train.shape}")
                st.write(f"y_test : {y_test.shape}")

            self.model_selection(model_type, self.selected_model, X_train, X_test, y_train, y_test)

        except Exception as e:
            st.error(f"Erreur lors de l'affichage du preprocessing: " + str(e))

    def model_selection(self, model_type, selected_model, X_train, X_test, y_train, y_test):
        use_pca = st.sidebar.toggle("Utiliser PCA ?", value=False)
        is_auto = st.sidebar.toggle("Recherche automatique ?", value=True)
        try:
            model = Modelisation(selected_model, X_train, X_test, y_train, y_test, use_pca, is_auto)
            st.title("Evaluation du Modèle")
            if model_type == "Régression":
                st.subheader("Résultat du Modèle :")
                kfold = st.sidebar.number_input("Choix du kfold (validation croisée) :", min_value=5,
                                                help="La valeur doit être un entier positif")
                if is_auto is True:
                    model.regressor_choice(kfold)
                else:
                    expander = st.sidebar.expander("Choix des hyperparamètres :")
                    regression_algorithms = DictModels.get_regressors_dict()
                    model_name = list(regression_algorithms["algorithme"][selected_model].keys())[0]
                    dict_params = {}
                    try:
                        if not regression_algorithms["algorithme"][selected_model][model_name]:
                            st.sidebar.info(f"Pas d\'hyperparamètres à choisir pour ce modèle.")
                        else:
                            for param, value in regression_algorithms["algorithme"][selected_model][model_name].items():
                                if isinstance(value[0], float) or isinstance(value[0], int):
                                    min_val = min(value)
                                    max_val = max(value)
                                    user_input = expander.number_input(param, min_value=min_val, max_value=max_val, value=min_val)
                                else:
                                    user_input = expander.selectbox(param, value)
                                dict_params[param] = user_input
                    except Exception as e:
                        st.error(f"Erreur lors de la récupération des paramètres: " + str(e))
                    try:
                        model.regressor(dict_params, kfold)
                    except Exception as e:
                        st.error(f"Erreur lors de la modélisation (manuelle): " + str(e))

                metrics_to_display = ["reg_metrics", "regression_plot", "regression_coefficient_hist",
                                      "courbe_apprentissage"]
                for metric in metrics_to_display:
                    if metric == "reg_metrics":
                        json_filename = "info_metrics.json"
                        r2, rmse, mae, score_train, score_test = model.reg_metrics(is_auto)
                        data = {
                            'Métrique': ['R²', 'RMSE', 'MAE'],
                            'Valeur': [r2, rmse, mae]
                        }
                        df_metrics = pd.DataFrame(data)
                        st.write("Métriques de performance :")
                        st.data_editor(df_metrics, hide_index=True)
                        metric_info = get_metrics_info(json_filename, metric)
                        accuracy = {
                            "Echantillon d'entraînement": ['Train', 'Test'],
                            'Score MSE': [(score_train), (score_test)]
                        }
                        st.write(metric_info)
                        df_accuracy = pd.DataFrame(accuracy)
                        st.write("Score de train et de test:")
                        st.data_editor(df_accuracy, hide_index=True)
                    elif metric == "regression_plot":
                        metric_info = get_metrics_info(json_filename, metric)
                        st.plotly_chart(model.regression_plot())
                        st.write(metric_info)
                    elif metric == "regression_coefficient_hist":
                        metric_info = get_metrics_info(json_filename, metric)
                        try:
                            st.plotly_chart(model.regression_coefficient_hist(selected_model, is_auto))
                            st.write(metric_info)
                        except:
                            st.warning(
                                f"Cette évaluation de modèle n'a pas d'intérêt pour ce modèle: {selected_model}.")
                    elif metric == "courbe_apprentissage":
                        metric_info = get_metrics_info(json_filename, metric)
                        st.plotly_chart(model.courbe_apprentissage(model_type, is_auto))
                        st.write(metric_info)

            else:
                st.subheader("Résultat du Modèle :")
                kfold = st.sidebar.number_input("Choix du kfold (validation croisée) :", min_value=5,
                                                help="La valeur doit être un entier positif")
                if is_auto is True:
                    model.classifier_choice(kfold)
                else:
                    expander = st.sidebar.expander("Choix des hyperparamètres :")
                    classification_algorithms = DictModels.get_classifiers_dict()
                    model_name = list(classification_algorithms["algorithme"][selected_model].keys())[0]
                    dict_params = {}
                    try:
                        if not classification_algorithms["algorithme"][selected_model][model_name]:
                            st.sidebar.info(f"Pas d\'hyperparamètres à choisir pour ce modèle.")
                        else:
                            for param, value in classification_algorithms["algorithme"][selected_model][model_name].items():
                                if isinstance(value[0], float) or isinstance(value[0], int):
                                    min_val = min(value)
                                    max_val = max(value)
                                    user_input = expander.number_input(param, min_value=min_val, max_value=max_val,
                                                                       value=min_val)
                                else:
                                    user_input = expander.selectbox(param, value)
                                dict_params[param] = user_input
                    except Exception as e:
                        st.error(f"Erreur lors de la récupération des paramètres: " + str(e))
                    try:
                        model.classifier(dict_params, kfold)
                    except Exception as e:
                        st.error(f"Erreur lors de la modélisation (manuelle): " + str(e))

                metrics_to_display = ["matrix_confusion", "report_classif", "roc_auc", "courbe_apprentissage"]
                json_filename = "info_metrics.json"
                for metric in metrics_to_display:
                    if metric == "matrix_confusion":
                        metric_info = get_metrics_info(json_filename, metric)
                        st.plotly_chart(model.matrix_confusion())
                        st.write(metric_info)
                    elif metric == "report_classif":
                        metric_info = get_metrics_info(json_filename, metric)
                        st.write("**Rapport de classification**")
                        st.write(model.report_classif())
                        st.write(metric_info)
                    elif metric == "roc_auc":
                        metric_info = get_metrics_info(json_filename, metric)
                        st.plotly_chart(model.roc_auc(selected_model, is_auto))
                        st.write(metric_info)
                    elif metric == "courbe_apprentissage":
                        metric_info = get_metrics_info(json_filename, metric)
                        st.plotly_chart(model.courbe_apprentissage(model_type, is_auto))
                        st.write(metric_info)

        except Exception as e:
            st.error(f"Erreur lors de la visualisation : {str(e)}")


if __name__ == '__main__':
    app = MachineLearningApp()
    app.run()