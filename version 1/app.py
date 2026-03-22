from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




app = Flask(__name__)

# train the model

df = pd.read_csv('Housing.csv')
x = df[['area', 'bedrooms', 'bathrooms', 'stories']]
y = df[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=32)

model = LinearRegression()
model.fit(x_train, y_train)

@app.route('/', methods=['POST', 'GET'])
def pricing():

    if request.method == 'POST':
        area = request.form['area']
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        stories = request.form['stories']
        if not area:
            area = 2000

        data = [{
            'area': area,
            'bedrooms': bedrooms,  
            'bathrooms': bathrooms,
            'stories': stories
        }]

        pred = pd.DataFrame(data)

        result = int(model.predict(pred)[0][0])

        return render_template('index.html', result=result)
    return render_template('index.html')

# from this model we concluse that bathrooms are the most factor in pricing and I don't know why

if __name__ == '__main__':
    app.run(debug=True)



