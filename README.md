# mobile-plan-classification-ml
Machine learning project that builds classification models to predict optimal telecom mobile plans (Smart vs Ultra) based on customer usage behavior using Python, Pandas, and Scikit-Learn.

## Project Overview

Telecommunication companies offer multiple mobile plans designed for different types of users. This project builds a machine learning model that predicts whether a customer should be assigned to the **Smart** or **Ultra** plan based on their monthly usage behavior.

The goal is to analyze usage patterns and build a classification model that recommends the most appropriate mobile plan.

---

## Dataset

The dataset contains monthly usage statistics for telecom customers.

Features include:

- calls — number of calls made
- minutes — total call minutes
- messages — number of SMS messages sent
- mb_used — internet data usage (MB)

Target variable:

- **is_ultra**
  - 0 = Smart plan
  - 1 = Ultra plan

---

## Project Workflow

1. Load and explore the dataset  
2. Split data into training, validation, and test sets  
3. Train multiple machine learning models  
4. Compare model performance  
5. Select the best model  
6. Evaluate the model on the test dataset  

---

## Models Tested

- Decision Tree
- Random Forest
- Logistic Regression

---

## Results

Best Model: Random Forest

Example performance:

Validation Accuracy: 0.81  
Test Accuracy: 0.80  

---

## Project Structure

mobile-plan-classification-ml

data/ – dataset files  
notebooks/ – exploratory analysis  
src/ – model training and evaluation scripts  
models/ – saved trained models  
results/ – evaluation outputs  

---

## Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/YOUR_USERNAME/mobile-plan-classification-ml.git
cd mobile-plan-classification-ml
pip install -r requirements.txt
