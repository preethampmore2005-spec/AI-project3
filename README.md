# Mall Customer Segmentation API

## Project Overview

This project demonstrates the deployment of a Machine Learning model using FastAPI and Docker.

The model predicts whether a customer belongs to a High Value Customer segment or a Low Value Customer segment based on:

* Gender
* Age
* Annual Income
* Spending Score

The API accepts customer information as input and returns the predicted customer segment.

---

## Dataset

Dataset Used: Mall Customers Dataset

Features:

* Genre
* Age
* Annual_Income_(k$)
* Spending_Score

Target:

* Segment = 1 (High Value Customer)
* Segment = 0 (Low Value Customer)

The target variable is automatically generated using:

```python
df["Segment"] = (df["Spending_Score"] >= 50).astype(int)
```

---

## Technologies Used

* Python
* Scikit-Learn
* FastAPI
* Uvicorn
* Docker
* Pandas
* NumPy

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python train_model.py
```

Output:

```text
Accuracy: 1.0
Model saved successfully!
```

This generates:

```text
model.pkl
```

---

## Run the API Locally

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /predict

Request:

```json
{
  "gender": 1,
  "age": 25,
  "annual_income": 60,
  "spending_score": 80
}
```

Response:

```json
{
  "segment": 1,
  "segment_name": "High Value Customer"
}
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t mall-customer-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 mall-customer-api
```

Open:

```text
http://localhost:8000/docs
```

---

## Project Structure

```text
Mall_Customer_API/
â
âââ Mall_Customers.csv
âââ train_model.py
âââ app.py
âââ requirements.txt
âââ Dockerfile
âââ README.md
âââ model.pkl
```

---

## Conclusion

This project successfully deploys a machine learning model as a REST API using FastAPI and Docker. The API can be used to predict customer segments based on demographic and spending characteristics and demonstrates the complete workflow from model training to deployment.