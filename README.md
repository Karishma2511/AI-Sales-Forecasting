# AI-Powered Sales Forecasting Platform

## 📌 Project Overview

The AI-Powered Sales Forecasting Platform is a machine learning-based solution designed to predict future sales for retail businesses. This project leverages FastAPI for real-time API deployment, XGBoost and LSTM for advanced time series forecasting, and Databricks & AWS Glue for data processing and transformation. The model is trained using Walmart's historical sales data stored in Amazon S3, providing insights that help businesses optimize inventory management and demand planning.

## 🛠️ Tech Stack

- **Backend & Deployment**: FastAPI | Render  
- **Machine Learning Models**: XGBoost | LSTM | Scikit-Learn  
- **Data Processing & Storage**: Databricks | AWS Glue | Amazon S3  
- **Libraries Used**: NumPy | Pandas | Matplotlib | Joblib | PySpark  

## 🚀 Features

✅ **Real-time Sales Prediction**: Predicts future sales using trained machine learning models.  
✅ **Advanced Forecasting Models**: Utilizes XGBoost and LSTM to improve accuracy.  
✅ **Cloud-Based Data Processing**: Implements AWS Glue & Databricks for ETL workflows.  
✅ **Seamless API Deployment**: FastAPI-based REST API hosted on Render.  
✅ **Feature Engineering**: Includes lag features, moving averages, and time-based transformations for enhanced predictions.  
✅ **Hyperparameter Optimization**: Improves model performance using GridSearch and Bayesian Optimization.  

## 📂 Project Workflow

### 1️⃣ Data Collection & Storage

- **Dataset**: Walmart Sales Dataset (Kaggle)  
- **Storage**: Amazon S3 (`s3://walmart-sales-dataset/raw/Walmart.csv`)  
- **ETL Processing**: AWS Glue transforms raw data into Parquet format (`s3://walmart-sales-dataset/processed/`)  

### 2️⃣ Data Processing & Feature Engineering

- **Databricks for EDA**: Missing values handling, trend analysis, seasonality detection.  
- **Feature Engineering**:  
  - Created lag features (`Sales_Lag_1`, `Sales_Lag_7`, `Sales_Lag_30`)  
  - Generated moving averages (`Sales_MA_7`, `Sales_MA_30`)  
  - Extracted time-based features (`Year`, `Month`, `WeekOfYear`, `DayOfWeek`)  
  - Added `IsWeekend` flag to capture weekend effects.  

### 3️⃣ Model Training & Optimization

- **Models Used**: XGBoost & LSTM (Long Short-Term Memory Networks)  
- **Hyperparameter Tuning**:  
  - Used GridSearchCV & Bayesian Optimization for tuning.  
  - Evaluated performance based on RMSE (Root Mean Squared Error).  
- **Model Selection**: XGBoost was chosen due to optimized RMSE.  

### 4️⃣ Model Deployment

- **Deployment**: Hosted on Render using FastAPI.  
- **API Endpoints**:  
  - `GET /` → API Health Check.  
  - `POST /predict` → Accepts JSON input and returns sales predictions.  

## 📌 API Usage

### 1️⃣ Start the FastAPI Server Locally

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2️⃣ API Endpoints

#### 1. Health Check

**GET /**  

**Response:**  

```json
{
    "message": "AI Sales Forecasting API is running!"
}
```

#### 2. Sales Prediction

**POST /predict**  
**Content-Type: application/json**  

```json
{
    "store_id": 1,
    "product_id": 101,
    "past_sales": [100, 120, 110, 105]
}
```

**Response:**  

```json
{
    "store_id": 1,
    "product_id": 101,
    "forecast": 115.3
}
```

## 📦 Installation & Setup

### 1️⃣ Clone Repository

```sh
git clone https://github.com/Karishma2511/AI-Sales-Forecasting.git
cd AI-Sales-Forecasting
```

### 2️⃣ Set Up Virtual Environment

```sh
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 4️⃣ Run FastAPI Server

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

📌 **Contributors:** [Karishma](https://github.com/Karishma2511)  
📌 **License:** MIT License  
