# 🎬 IMDb Sentiment Analysis App | Dockerized End-to-End ML Project

<p align="center">


🚀 FastAPI • Streamlit • Scikit-learn • Docker • MLOps Mini Project

</p>

---

## 📌 Overview

This project is an end-to-end Machine Learning application that predicts whether a movie review is:

✅ Positive  
❌ Negative

It combines:

- 🧠 Machine Learning model (Logistic Regression + TF-IDF)
- ⚡ FastAPI backend for inference
- 🎨 Streamlit frontend for interactive analysis
- 🐳 Dockerized multi-container deployment with Docker Compose

Supports:

✔ Single Review Prediction  
✔ Bulk Review Analysis  
✔ Confidence Scoring  
✔ Aggregate Sentiment Summary

---

## 🏗 Project Architecture

```text
User Input (Streamlit UI)
        ↓
 FastAPI REST API
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorizer
        ↓
Logistic Regression Model
        ↓
Sentiment Prediction
        ↓
Results Returned to UI
```

---

## ⚙ Tech Stack

### 🧠 Machine Learning
- Python
- Scikit-learn
- Logistic Regression
- TF-IDF

### 🚀 Backend
- FastAPI
- Uvicorn

### 🎨 Frontend
- Streamlit

### 🐳 Deployment / MLOps
- Docker
- Docker Compose

---

## ✨ Features

### 🔹 Single Review Analysis

- Predict sentiment
- Confidence score
- Instant inference

---

### 🔹 Bulk Review Analysis

- Batch inference
- Positive vs Negative counts
- Average confidence
- Audience reception summary

---

## 📡 API Endpoints

```http
GET /health
POST /predict
POST /predict/bulk
```

---

# 🛠 How I Built This

--- Built an NLP preprocessing pipeline for cleaning and preparing text data  
- Trained a sentiment classification model using TF-IDF and Logistic Regression  
- Developed a FastAPI backend to serve model predictions  
- Created a Streamlit frontend for interactive user analysis  
- Containerized the full application using Docker and Docker Compose---

# 🚧 Key Challenges

- Environment setup and dependency configuration  
- Integrating ML model, API, frontend, and containers  
- Multi-container Docker orchestration  
- Debugging runtime and communication issues  
- Iterative testing and deployment refinement


# 📂 Project Structure

```text
.
├── api.py
├── streamlit_app.py
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── LR_model.pkl
├── LR_vectorizer.pkl
└── README.md
```

---

# 🚀 Run Locally

## Clone

```bash
git clone <your_repo_url>
cd sentiment_analysis
```

---

## Run With Docker

```bash
docker compose up --build
```

---

# 🌐 Access App

### FastAPI Docs

```text
http://localhost:8000/docs
```

---

### Streamlit App

```text
http://localhost:8501
```

---

## 🎯 Sample Prediction

Input:

```text
This movie was amazing and brilliantly acted.
```

Output:

```text
POSITIVE
Confidence: 97%
```

---

# 📈 Future Improvements

- 🤖 BERT / LLM-based sentiment model
- ☁ Cloud deployment
- 🔄 CI/CD pipeline
- 📊 Model monitoring
- ☸ Kubernetes deployment

---

# 🔴 Live Demo

Add deployed link:

```text
[
https://sentimentanalysisproject-h3fwfhg8mcxplawbeukfea.streamlit.app/]
```

---

# 💻 GitHub Repo

```text
[https://github.com/ranganath18/Sentiment_Analysis_project/blob/main/api.py]
```

---

# 📚 Key Learnings

This project improved my understanding of:

✔ NLP preprocessing  
✔ Model serving  
✔ API development  
✔ Docker containerization  
✔ Multi-container architecture  
✔ Introductory MLOps workflow

---

## 🧑‍💻 Author

**Rangam Ranganath
B.TECH CSE(DATA SCIENCE) **

🔗 LinkedIn: [www.linkedin.com/in/ranganath-rangam-49a2a324a]  
💻 GitHub: [https://github.com/ranganath18]

---

## ⭐ If you found this project interesting, consider starring the repo."# Sentiment_Analysis_project" 
