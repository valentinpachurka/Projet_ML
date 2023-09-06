import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import sqlalchemy


class DataPreprocessor:
    def __init__(self, selected_table, engine, split_ratio):
        self.dataframe = None
        self.original_dataframe = None
        self.selected_table = selected_table
        self.engine = engine
        self.split_ratio = split_ratio
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self, selected_columns):
        self.load_data()
        self.dataframe = self.dataframe_columns(selected_columns)
        self.val_nan()
        self.outliers()
        self.encodage()
        self.standardization()
        self.split_data()


    def load_data(self):
        dataframe = pd.read_sql_query(f"SELECT * FROM {self.selected_table}", self.engine, index_col="id")
        self.dataframe = dataframe
        self.original_dataframe = self.dataframe.copy()
        return self.dataframe

    def dataframe_columns(self, columns):
        self.selected_columns = self.dataframe[columns]
        return self.selected_columns

    def val_nan(self):
        total_rows = len(self.dataframe)
        for column in self.dataframe.columns:
            nan_count = self.dataframe[column].isna().sum()
            nan_percentage = (nan_count / total_rows) * 100
            if nan_percentage > 25:
                self.dataframe.drop(column, axis=1, inplace=True)
        self.dataframe.dropna(inplace=True)

    def outliers(self):
        for i in self.dataframe.columns:
            if i != 'target':
                if self.dataframe[i].dtype != 'object':
                    median = self.dataframe[i].median()
                    std = self.dataframe[i].std()
                    limite_inferieur = median - 3 * std
                    limite_superieur = median + 3 * std
                    self.dataframe = self.dataframe.loc[
                        (self.dataframe[i] >= limite_inferieur) & (self.dataframe[i] <= limite_superieur)]

    def encodage(self):
        for i in self.dataframe.columns:
            le = LabelEncoder()
            self.dataframe[i] = le.fit_transform(self.dataframe[i])

    def standardization(self):
        scaler = StandardScaler()
        self.dataframe.loc[:, self.dataframe.columns != 'target'] = scaler.fit_transform(
            self.dataframe.loc[:, self.dataframe.columns != 'target'])
        return self.dataframe

    def split_data(self):
        try:
            X = self.dataframe.drop('target', axis=1)
            y = self.dataframe['target']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.split_ratio,
                                                                                    random_state=42)
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            st.error("Erreur lors de la récupération de target: " + str(e))