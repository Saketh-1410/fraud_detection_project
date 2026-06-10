import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("creditcard.csv")

# ===============================
# 2. Separate features and target
# ===============================
X = df.drop("Class", axis=1)
y = df["Class"]

print("Class distribution BEFORE balancing:")
print(y.value_counts())

# ===============================
# 3. Train-test split (IMPORTANT)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. Feature scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Handle imbalance using SMOTE
#    (ONLY on training data)
# ===============================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# ===============================
# 6. Train Random Forest model
# ===============================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

# ===============================
# 7. Model evaluation
# ===============================
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 8. Save model and scaler
# ===============================
pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel trained using proper data science imbalance handling")
