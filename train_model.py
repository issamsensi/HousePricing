import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle
from pathlib import Path


model_path = Path('model.pkl')


def train_and_save(model_path=model_path):
    df = pd.read_csv('Housing.csv')

    x = df.drop(columns=['price'])
    y = df['price']

    categorical_features = [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea',
        'furnishingstatus',
    ]

    numerical_features = [
        'area',
        'bedrooms',
        'bathrooms',
        'stories',
        'parking',
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression()),
        ]
    )

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.1, random_state=32)

    model.fit(x_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

if __name__ == '__main__':
    train_and_save()