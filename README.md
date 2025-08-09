<img width="1916" height="960" alt="cars" src="https://github.com/user-attachments/assets/641217ad-f73e-49a2-b64e-46464b5cee96" />





# 🚗 Cars Price Prediction App (2025 Dataset)

This is a **Machine Learning + Streamlit** based web application that predicts the price of a car using the latest 2025 cars dataset.  
Users simply input the car specifications such as engine size, horsepower, top speed, fuel type, and other features, and the app instantly predicts the car price.

---

## 📌 Features

- **Trained on 2025 Cars Dataset** for modern and relevant predictions
- **Random Forest Regressor** for high accuracy
- Automatic encoding for categorical features like Fuel Type
- Simple, clean, and responsive Streamlit UI
- Ready for **local use** and **online deployment**

---

## 🛠 Tech Stack

- **Python 3.10**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**

---

## 📂 Project Structure

```
├── app.py                # Frontend Streamlit code
├── backend.py            # Backend training & prediction logic
├── cars_dataset_2025.csv # Training dataset
├── car_price_model.pkl   # Trained ML model
├── label_encoders.pkl    # Encoders for categorical features
└── requirements.txt      # Dependencies list
```

---

## 🚀 How to Run Locally

1️⃣ Clone the repository:

```bash
git clone https://github.com/FarazhussainAI250/cars-price-prediction.git
cd cars-price-prediction
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app:

```bash
streamlit run app.py
```

---
