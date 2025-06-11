## 📈 AI Stock Advisor App

An intelligent stock recommendation platform built with **Streamlit** that helps users discover personalized stock picks based on their risk appetite, sector preferences, and financial indicators. It uses **machine learning models (XGBoost + ExtraTrees)** and real-time data from Yahoo Finance to generate reliable suggestions.

---

### 🚀 Features

* 🔐 **User Authentication System** (Login/Signup with SQLite)
* ⚖️ **Risk-Based Recommendations** (Conservative, Moderate, Aggressive)
* 📊 **Technical & Fundamental Indicators**: RSI, MACD, Volatility, P/E, Dividend Yield, etc.
* 🧠 **ML Ensemble Prediction** using `XGBoost` and `ExtraTreesClassifier`
* 📈 **Dynamic Filtering**: Sector, PE ratio, Dividend Yield, Investment Horizon
* ✅ **Fallback Scoring System** when ML models cannot be trained
* 📉 **Historical Analysis** with Moving Averages, RSI, MACD, etc.

---

### 📸 Demo Preview

> ![fullscreen](https://github.com/user-attachments/assets/0ec9bd35-522a-42f5-8142-4d39b50d4fc7)


---

### 🧪 Tech Stack

* **Frontend & App Interface**: Streamlit
* **Backend/Storage**: SQLite (for user authentication)
* **Data Source**: Yahoo Finance API
* **Machine Learning**:

  * XGBoost Classifier
  * Extra Trees Classifier
* **Libraries**:

  * pandas, numpy, scikit-learn, streamlit, yfinance, matplotlib, seaborn, easyocr, etc.

---

### ⚙️ Installation & Setup

#### 📌 Pre-requisites:

* Python 3.8–3.11 recommended
* pip or conda

#### 💻 Local Setup:

```bash
# Clone the repo
[git clone https://github.com/your-username/ai-stock-recommender.git](https://github.com/Siddhesh1732/Stock-Recommender.git)
cd Stock-Recommender

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

### ✅ Example Login Credentials

> On first run, create a new account under the **Sign Up** tab with your details and preferences.

---


### 📜 License

MIT License — free to use, modify, and distribute.

---
