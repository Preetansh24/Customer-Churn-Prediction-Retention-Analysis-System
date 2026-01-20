# 📊 Customer Churn Prediction App

A visually stunning Streamlit web application for predicting customer churn using Machine Learning.

## ✨ Features

- **Customer Input Form** - Comprehensive form with all 19 customer attributes
- **Churn Probability** - Real-time probability prediction displayed as percentage
- **Risk Category** - Color-coded badges (High/Medium/Low)
- **Key Factors** - Top 5 factors influencing the prediction
- **Recommendations** - Personalized retention strategies

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (First time only)
```bash
python model_trainer.py
```

### 3. Run the App
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
churn_app/
├── app.py                 # Streamlit application
├── model_trainer.py       # Model training script
├── requirements.txt       # Python dependencies
└── model/
    ├── churn_model.pkl    # Trained Random Forest model
    └── feature_names.pkl  # Feature names for importance
```

## 🎯 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 19 customer attributes including tenure, contract type, services, and charges
- **Training Data**: Telco Customer Churn dataset (7,043 customers)

## 📸 Screenshots

The app features a modern dark theme with:
- Glassmorphism effects
- Animated probability gauge
- Color-coded risk badges
- Interactive recommendations

## 📝 License

MIT License
