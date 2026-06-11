# 🎓 School Dropout Prediction System

A Machine Learning-powered application designed to predict whether a student is at risk of dropping out based on academic, demographic, and socioeconomic factors. This project aims to help educational institutions identify at-risk students early and implement targeted interventions to improve retention and academic success.

---

## 📖 Overview

Student dropout is a significant challenge for educational institutions worldwide. By leveraging machine learning techniques, this project analyzes student-related data and predicts the likelihood of dropout, enabling proactive support and decision-making.

### Key Objectives

- Predict student dropout risk using machine learning models.
- Assist institutions in identifying at-risk students.
- Support data-driven educational interventions.
- Provide an easy-to-use interface for predictions.

---

## ✨ Features

- 📊 Student dropout prediction
- 🧹 Data preprocessing pipeline
- 🤖 Machine Learning model training and evaluation
- ⚡ Real-time prediction generation
- 💾 Saved and reusable trained models
- 🌐 Web-based interface for user interaction
- 📈 Scalable architecture for future enhancements

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-Learn
- Pandas
- NumPy

### Web Framework
- Flask

### Data Handling
- Pandas
- NumPy

### Model Persistence
- Pickle
- Joblib

### Frontend
- HTML
- CSS

---

## 📂 Project Structure

```text
School_dropout_prediction/
│
├── Data/                  # Dataset files
├── models/                # Trained machine learning models
├── src/                   # Source code and utilities
├── app/                   # Application modules
│
├── main.py                # Main application entry point
├── predict.py             # Prediction logic
├── requirements.txt       # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Git
- pip

### Clone the Repository

```bash
git clone https://github.com/codenano077/School_dropout_prediction.git
cd School_dropout_prediction
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the application:

```bash
python main.py
```

Once the application is running, open your browser and visit:

```text
http://localhost:5000
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Collection

Gather student-related information including:

- Academic performance
- Attendance records
- Demographic information
- Socioeconomic indicators

### 2. Data Preprocessing

- Handle missing values
- Encode categorical variables
- Feature scaling
- Data cleaning

### 3. Model Training

Train machine learning models using historical student data.

### 4. Model Evaluation

Evaluate model performance using:

- Accuracy
- Precision
- Recall
- F1 Score

### 5. Model Deployment

Deploy the trained model through a web interface for real-time predictions.

---

## 📊 Prediction Workflow

```text
User Input
     │
     ▼
Data Preprocessing
     │
     ▼
Trained ML Model
     │
     ▼
Prediction Result
     │
     ▼
Dropout Risk Assessment
```

---

## 🎯 Sample Use Case

### Input

```text
Age: 19
Attendance: 65%
Previous Grades: Average
Parental Education: Secondary
Financial Support: No
```

### Output

```text
Prediction: High Risk of Dropout
Confidence Score: 87%
```

---

## 📈 Potential Applications

- Educational institutions
- Student retention programs
- Academic counseling systems
- Government education initiatives
- Educational research projects

---

## 🔮 Future Enhancements

- Interactive analytics dashboard
- Multiple model comparison
- Explainable AI (XAI) integration
- Cloud deployment using AWS/Azure
- Student performance tracking
- Real-time database integration
- Mobile-friendly interface
- Automated reporting system

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 🧪 Future Model Improvements

Possible algorithms to experiment with:

- Random Forest
- XGBoost
- LightGBM
- Support Vector Machines
- Neural Networks

---

## 👨‍💻 Author

### Darshan Raghuram Shetty

Aspiring Software Developer and Machine Learning Enthusiast.

- GitHub: https://github.com/codenano077

---

## ⭐ Support

If you found this project useful, consider giving it a star ⭐ on GitHub.

```bash
⭐ Star this repository
```

---

## 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.

---

### Repository Link

https://github.com/codenano077/School_dropout_prediction
