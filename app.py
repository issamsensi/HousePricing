import pickle
from pathlib import Path
import pandas as pd

from flask import Flask, render_template, request

from train_model import train_and_save


MODEL_PATH = Path('model.pkl')


app = Flask(__name__)


def get_default_form_data():
    return {
        'area': 2000,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 1,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 1,
        'prefarea': 'no',
        'furnishingstatus': 'semi-furnished',
    }


def build_features_frame(form_data):
    return pd.DataFrame([{
        'area': form_data['area'],
        'bedrooms': form_data['bedrooms'],
        'bathrooms': form_data['bathrooms'],
        'stories': form_data['stories'],
        'mainroad': form_data['mainroad'],
        'guestroom': form_data['guestroom'],
        'basement': form_data['basement'],
        'hotwaterheating': form_data['hotwaterheating'],
        'airconditioning': form_data['airconditioning'],
        'parking': form_data['parking'],
        'prefarea': form_data['prefarea'],
        'furnishingstatus': form_data['furnishingstatus'],
    }])


def is_model_compatible(candidate_model):
    try:
        defaults = get_default_form_data()
        sample = build_features_frame(defaults)
        candidate_model.predict(sample)
        return True
    except Exception:
        return False


def load_model():
    if not MODEL_PATH.exists():
        return train_and_save()

    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    if model and is_model_compatible(model):
        return model

    return train_and_save()


model = load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    default_form_data = get_default_form_data()

    if request.method == 'POST':
        try:
            area = float(request.form.get('area', default_form_data['area']) or default_form_data['area'])
            bedrooms = int(request.form.get('bedrooms', default_form_data['bedrooms']) or default_form_data['bedrooms'])
            bathrooms = int(request.form.get('bathrooms', default_form_data['bathrooms']) or default_form_data['bathrooms'])
            stories = int(request.form.get('stories', default_form_data['stories']) or default_form_data['stories'])
            parking = int(request.form.get('parking', default_form_data['parking']) or default_form_data['parking'])

            mainroad = request.form.get('mainroad', default_form_data['mainroad'])
            guestroom = request.form.get('guestroom', default_form_data['guestroom'])
            basement = request.form.get('basement', default_form_data['basement'])
            hotwaterheating = request.form.get('hotwaterheating', default_form_data['hotwaterheating'])
            airconditioning = request.form.get('airconditioning', default_form_data['airconditioning'])
            prefarea = request.form.get('prefarea', default_form_data['prefarea'])
            furnishingstatus = request.form.get('furnishingstatus', default_form_data['furnishingstatus'])

            binary_values = {'yes', 'no'}
            furnishing_values = {'furnished', 'semi-furnished', 'unfurnished'}

            if (
                mainroad not in binary_values
                or guestroom not in binary_values
                or basement not in binary_values
                or hotwaterheating not in binary_values
                or airconditioning not in binary_values
                or prefarea not in binary_values
                or furnishingstatus not in furnishing_values
            ):
                raise ValueError
        except ValueError:
            return render_template(
                'index.html',
                error='Please provide valid values for all fields.',
                form_data=default_form_data,
            )

        form_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'parking': parking,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus,
        }

        pred = build_features_frame(form_data)
        result = int(model.predict(pred)[0])

        return render_template('index.html', result=result, form_data=form_data)

    return render_template(
        'index.html',
        form_data=default_form_data,
    )


if __name__ == '__main__':
    app.run(debug=True)