# Imports pour la classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Imports pour la régression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, mean_absolute_error, r2_score, \
    mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st


class DictModels:

    def __init__(self):
        pass

    @staticmethod
    def get_classifiers_dict():
        dictionnaire_classifiers = {
            "algorithme": {
                "Régression Logistique": {
                    "LogisticRegression()": {
                        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'newton-cholesky', 'sag', 'saga'],
                        "penalty": ['l1', 'l2', 'elasticnet', None],
                        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
                },
                "Arbres de Décision": {
                    "DecisionTreeClassifier()": {'max_depth': [2, 3, 5, 10, 20],
                                                'min_samples_leaf': [5, 10, 20, 50, 100],
                                                 'criterion': ['poisson', 'squared_error', 'friedman_mse', 'absolute_error'],
                                                 'max_features': ["sqrt", "log2"]}
                },
                "Forêts aléatoires": {
                    "RandomForestClassifier()": {"n_estimators": [10, 100, 1000],
                                                 'criterion': ['poisson', 'squared_error', 'friedman_mse', 'absolute_error'],
                                                 "max_features": ['sqrt', 'log2', None]}
                },
                "SVM (Support Vector Machines)": {
                    "SVC()": {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                              "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                              "gamma": ['scale', 'auto']}
                },
                "Réseau de Neurones": {
                    "MLPClassifier()": {"hidden_layer_sizes": [(100,), (50, 50), (50, 25, 10)],
                                        "activation": ['relu', 'logistic', 'tanh', 'identify'],
                                        "solver": {'lbfgs', 'sgd', 'adam'},
                                        "alpha": [0.0001, 0.001, 0.01],
                                        "learning_rate": {'constant', 'invscaling', 'adaptive'}}
                },
                "Naïve Bayes": {
                    "GaussianNB()": {}
                },
                "Gradient Boosting": {
                    "GradientBoostingClassifier()": {"n_estimators": [10, 100, 1000],
                                                     "learning_rate": [0.001, 0.01, 0.1],
                                                     "max_depth": [3, 7, 9],
                                                     "loss": ['absolute_error', 'squared_error', 'quantile', 'huber'],
                                                     "criterion": ['friedman_mse', 'squared_error']}
                },
                "LDA (Linear Discriminant Analysis)": {
                    "LinearDiscriminantAnalysis()": {"solver": ['svd', 'lsqr', 'eigen']}
                },
                "XGBoost": {
                    "XGBClassifier()": {"n_estimators": [10, 100, 1000],
                                        "learning_rate": [0.001, 0.01, 0.1],
                                        "max_depth": [3, 7, 9]}
                }
            }
        }

        return dictionnaire_classifiers

    @staticmethod
    def get_regressors_dict():
        dictionnaire_regressors = {
            "algorithme": {
                "Régression Linéaire": {
                    "LinearRegression()": {}
                },
                "Régression Ridge": {
                    "Ridge()": {"alpha": [1.0, 0.1, 0.01, 0.001, 0.0001],
                                "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga', 'lbfgs']}
                },
                "Régression Lasso": {
                    "Lasso()": {"alpha": [1.0, 0.1, 0.01, 0.001, 0.0001],
                                "selection": ["cyclic", "random"]}
                },
                "Régression ElasticNet": {
                    "ElasticNet()": {"alpha": [1.0, 0.1, 0.01, 0.001, 0.0001],
                                     "l1_ratio": np.arange(0, 1, 0.1),
                                     "selection": ["cyclic", "random"]}
                },
                "Arbres de Décision": {
                    "DecisionTreeRegressor()": {'max_depth': [2, 3, 5, 10, 20],
                                                'max_features': ['sqrt', 'log2', None],
                                                'min_samples_leaf': [5, 10, 20, 50, 100],
                                                'criterion': ['friedman_mse', 'absolute_error', 'poisson',
                                                              'squared_error']}
                },
                "Forêts aléatoires": {
                    "RandomForestRegressor()": {"n_estimators": [10, 100, 1000],
                                                "max_features": ['sqrt', 'log2', None],
                                                'criterion': ['friedman_mse', 'absolute_error', 'poisson',
                                                              'squared_error']}
                },
                "SVM (Support Vector Machines)": {
                    "SVR()": {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                              "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                              "gamma": ['scale', 'auto']}
                },
                "Réseau de Neurones": {
                    "MLPRegressor()": {"hidden_layer_sizes": [(100,), (50, 50), (50, 25, 10)],
                                       "activation": ['relu', 'logistic'],
                                       "alpha": [0.0001, 0.001, 0.01]}
                },
                "Gradient Boosting": {
                    "GradientBoostingRegressor()": {"n_estimators": [10, 100, 1000],
                                                    "learning_rate": [0.001, 0.01, 0.1],
                                                    "max_depth": [3, 7, 9]}
                },
                "XGBoost": {
                    "XGBRegressor()": {"n_estimators": [10, 100, 1000],
                                       "learning_rate": [0.001, 0.01, 0.1],
                                       "max_depth": [3, 7, 9]}
                }
            }
        }

        return dictionnaire_regressors


class Modelisation:

    def __init__(self, model, X_train, X_test, y_train, y_test, use_pca, is_auto):
        self.grid_search = None
        self.kfold = None
        self.is_auto = is_auto
        self.use_pca = use_pca
        self.model = model
        self.model_manu = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred_train = None
        self.y_pred_test = None

    def classifier_choice(self, kfold):
        classification_algorithms = DictModels.get_classifiers_dict()
        results = {}
        key = list(classification_algorithms["algorithme"][self.model].keys())[0]
        value = classification_algorithms["algorithme"][self.model][key]
        estimator = eval(key)
        self.grid_search = GridSearchCV(estimator, value, scoring='accuracy', cv=kfold)
        self.grid_search.fit(self.X_train, self.y_train)
        self.y_pred_train = self.grid_search.predict(self.X_train)
        self.y_pred_test = self.grid_search.predict(self.X_test)

        results[f"{self.model} - {key}"] = {
            "Meilleurs hyperparamètres": self.grid_search.best_params_,
            "Précision sur test": self.grid_search.best_estimator_.score(self.X_test, self.y_test)
        }
        results_df = pd.DataFrame.from_dict(results, orient='index')
        for index, row in results_df.iterrows():
            st.write(f"**{index}**")
            st.write(f"**Précision sur l'ensemble de test:** \t{row['Précision sur test']}")
            hyperparameters = row["Meilleurs hyperparamètres"]
            if isinstance(hyperparameters, dict):
                st.write("**Meilleurs hyperparamètres :**")
                for key, value in hyperparameters.items():
                    st.write(f"*{key}* => \t**{value}**")

        return results

    def classifier(self, dict_params, kfold):
        self.kfold = kfold
        classification_algorithms = DictModels.get_classifiers_dict()
        key = list(classification_algorithms["algorithme"][self.model].keys())[0]
        key = key.replace('(', '').replace(')', '')
        key_class = globals()[key]
        self.model_manu = key_class(**dict_params)
        scores = cross_val_score(self.model_manu, self.X_train, self.y_train, scoring='accuracy', cv=kfold, error_score='raise')
        self.model_manu.fit(self.X_train, self.y_train)
        self.y_pred_train = self.model_manu.predict(self.X_test)
        self.y_pred_test = self.model_manu.predict(self.X_test)
        st.write(f"**Précision sur la validation croisée:**\t{scores.mean()} sur {kfold} folds")
        return scores

    def regressor_choice(self, kfold):
        regression_algorithms = DictModels.get_regressors_dict()
        results = {}
        key = list(regression_algorithms["algorithme"][self.model].keys())[0]
        value = regression_algorithms["algorithme"][self.model][key]
        estimator = eval(key)
        if self.use_pca is True:
            pipeline = Pipeline([('pca', PCA()), ('regression', estimator)])
            param_grid = {'pca_n_components': np.arange(1, 15, 1)}
            param_grid.update(value)
            self.grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=kfold)
        else:
            self.grid_search = GridSearchCV(estimator, value, scoring='neg_mean_squared_error', cv=kfold)
        self.grid_search.fit(self.X_train, self.y_train)
        self.y_pred_train = self.grid_search.predict(self.X_train)
        self.y_pred_test = self.grid_search.predict(self.X_test)

        results[f"{self.model} - {key}"] = {
            "Meilleurs hyperparamètres": self.grid_search.best_params_,
            "Erreur MSE": self.grid_search.best_score_
        }
        results_df = pd.DataFrame.from_dict(results, orient='index')
        for index, row in results_df.iterrows():
            st.write(f"**{index}**")
            st.write(f"**Erreur MSE :** \t{row['Erreur MSE']}")
            hyperparameters = row["Meilleurs hyperparamètres"]
            if isinstance(hyperparameters, dict):
                st.write("**Meilleurs hyperparamètres :**")
                for key, value in hyperparameters.items():
                    st.write(f"*{key}* => \t**{value}**")

        return results

    def regressor(self, dict_params, kfold):
        self.kfold = kfold
        regression_algorithms = DictModels.get_regressors_dict()
        key = list(regression_algorithms["algorithme"][self.model].keys())[0]
        key = key.replace('(', '').replace(')', '')
        key_class = globals()[key]
        self.model_manu = key_class(**dict_params)
        scores = cross_val_score(self.model_manu, self.X_train, self.y_train, scoring='neg_mean_squared_error',
                                 cv=kfold, error_score='raise')
        self.reg = self.model_manu.fit(self.X_train, self.y_train)
        self.y_pred_train = self.model_manu.predict(self.X_test)
        self.y_pred_test = self.model_manu.predict(self.X_test)
        st.write(f"**Précision sur la validation croisée:**\t{scores.mean()} sur {kfold} folds")
        return scores

    def matrix_confusion(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_pred_test)
        fig = ff.create_annotated_heatmap(
            z=np.flip(conf_matrix, 0),
            x=['Classe Prédite 0', 'Classe Prédite 1'],
            y=['Classe Réelle 1', 'Classe Réelle 0'],
            colorscale='Blues',
        )
        fig.update_layout(
            title="Matrice de Confusion",
            xaxis_title="Classe Prédite",
            yaxis_title="Classe Réelle",
        )

        return fig

    def report_classif(self):
        class_report = classification_report(self.y_test, self.y_pred_test, output_dict=True)
        df = pd.DataFrame(class_report)
        return df

    def roc_auc(self, selected_model, is_auto):
        if is_auto is True:
            if selected_model == "SVM (Classification)":
                fpr, tpr, _ = roc_curve(self.y_test, self.grid_search.decision_function(self.X_test))
            else:
                fpr, tpr, _ = roc_curve(self.y_test, self.grid_search.predict_proba(self.X_test)[:, 1])
        else:
            if selected_model == "SVM (Classification)":
                fpr, tpr, _ = roc_curve(self.y_test, self.model_manu.decision_function(self.X_test))
            else:
                fpr, tpr, _ = roc_curve(self.y_test, self.model_manu.predict_proba(self.X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = {:.2f})'.format(roc_auc)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line_dash='dash', name='Random'))
        fig.update_layout(
            title='Courbe ROC',
            xaxis=dict(title='Taux de faux positifs'),
            yaxis=dict(title='Taux de vrais positifs'),
            showlegend=True
        )
        return fig

    def reg_metrics(self, is_auto):
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
        r2 = r2_score(self.y_test, self.y_pred_test)
        mae = mean_absolute_error(self.y_test, self.y_pred_test)
        if is_auto is True:
            score_train = self.grid_search.score(self.X_train, self.y_train)
            score_test = self.grid_search.score(self.X_test, self.y_test)
        else:
            score_train = self.model_manu.score(self.X_train, self.y_train)
            score_test = self.model_manu.score(self.X_test, self.y_test)
        return r2, rmse, mae, score_train, score_test

    def regression_plot(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.y_test, y=self.y_pred_test, mode='markers', name='Données', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[min(self.y_test), max(self.y_test)], y=[min(self.y_test), max(self.y_test)],
                                 mode='lines', name='Ligne de régression', line=dict(color='red', dash='dash')))
        fig.update_layout(
            xaxis_title='Valeurs réelles',
            yaxis_title='Prédictions',
            title='Diagramme de dispersion',
            showlegend=True,
            legend=dict(x=0.7, y=0.9),
        )
        return fig

    def regression_coefficient_hist(self, selected_model, is_auto):
        if is_auto is True:
            coefficients = self.grid_search.best_estimator_.coef_
        else:
            coefficients = self.reg.coef_
        feature_names = self.X_train.columns
        sorted_coefficients = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
        feature_names, sorted_coefficients = zip(*sorted_coefficients)

        fig = go.Figure(data=[go.Bar(x=feature_names, y=sorted_coefficients)])
        fig.update_layout(
            xaxis_title='Variable Explicative',
            yaxis_title='Coefficient de Régression (Trié par valeur absolue)',
            title=f'Histogramme des Coefficients de {selected_model} (Trié par valeur absolue)',
        )
        return fig

    def courbe_apprentissage(self, model_type, is_auto):
        if is_auto is True:
            best_estimator = self.grid_search.best_estimator_
            kfold = 5
        else:
            best_estimator = self.model_manu
            kfold = self.kfold
        if model_type == "Régression":
            train_sizes, train_scores, test_scores = learning_curve(
                best_estimator,
                self.X_train,
                self.y_train,
                cv=kfold,
                scoring='neg_mean_absolute_error',
                train_sizes=np.linspace(0.01, 1.0, 50)
            )
        else:
            train_sizes, train_scores, test_scores = learning_curve(
                best_estimator,
                self.X_train,
                self.y_train,
                cv=kfold,
                scoring='accuracy',
                train_sizes=np.linspace(0.01, 1.0, 50)
            )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode='lines+markers',
            name='Train',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode='lines+markers',
            name='Test',
            line=dict(color='green')
        ))
        fig.update_layout(
            title='Courbes d\'apprentissage',
            xaxis_title='Taille de l\'échantillon d\'entraînement',
            yaxis_title='Score',
        )
        return fig