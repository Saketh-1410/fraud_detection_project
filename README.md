# Fraud Detection Project

Machine-learning demo for online transaction fraud detection. The project includes:

- A Flask web app for checking individual transactions.
- A Streamlit dashboard demo.
- A Random Forest training script using SMOTE for class imbalance.

## Project Structure

```text
.
+-- fraud_website/
|   +-- app.py
|   +-- dashboard.py
|   +-- fraud_model.pkl
|   +-- scaler.pkl
|   +-- templates/
|       +-- index.html
+-- train_model.py
+-- requirements.txt
+-- README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Flask App

```bash
cd fraud_website
python app.py
```

Open the local URL shown in the terminal.

## Run The Dashboard

```bash
cd fraud_website
streamlit run dashboard.py
```

## Train The Model

Place `creditcard.csv` in the project root, then run:

```bash
python train_model.py
```

The dataset is intentionally ignored by Git because it is larger than GitHub's regular 100 MB file limit.
