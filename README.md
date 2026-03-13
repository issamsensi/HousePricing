# 🏠 House Pricing Predictor

A machine learning web application that predicts house prices based on key property features using **Linear Regression**.

---

## 📋 Overview

This project uses a trained Linear Regression model on housing data to estimate the price of a house. Users can input property details through a web interface and get an instant price prediction.

---

## ScreenShoot

![House Pricing Predictor](static/images/housepricing.png)

## 🚀 Features

- Predict house prices based on:
  - **Area** (sq ft)
  - **Number of Bedrooms**
  - **Number of Bathrooms**
  - **Number of Stories**
- Simple and clean web interface
- Real-time predictions powered by Flask

---

## 🛠️ Tech Stack

| Layer        | Technology              |
|--------------|-------------------------|
| Backend      | Python, Flask           |
| ML Model     | Scikit-learn (Linear Regression) |
| Data         | Pandas                  |
| Frontend     | HTML, CSS               |

---

## 📁 Project Structure

```
HousePricing/
├── app.py              # Flask application & ML model training
├── Housing.csv         # Dataset
├── predict.ipynb       # Jupyter notebook for data exploration
├── templates/
│   └── index.html      # Frontend UI
├── static/
│   └── style.css       # Styling
├── README.md
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/issamsensi/HousePricing.git
cd HousePricing
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## ⚠️ Disclaimer

> Predictions may have significant errors. This model is for educational purposes only and should not be used for real estate decisions.

---

## 📊 Dataset

The model is trained on `Housing.csv` using the following features:
- `area` — Total area of the house
- `bedrooms` — Number of bedrooms
- `bathrooms` — Number of bathrooms
- `stories` — Number of stories

---

## 🔗 Related Project

If you're interested in the data preprocessing workflow for `housing.csv`, check out this related project:  
[Data-Mining-Tp](https://github.com/issamsensi/Data-Mining-Tp.git)


## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

[Issam SENSI](https://github.com/issamsensi)

## Portfolio

[issamsensi.com](https://issamsensi.com)