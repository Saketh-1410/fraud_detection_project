from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Store recent predictions for dashboard (optional)
prediction_log = []


def generate_v_features(amount, time):
    """
    Simulates backend generation of PCA-like features (V1-V28)
    """
    v = np.zeros(28)

    v[0]  = amount / 1000
    v[1]  = np.log1p(amount)
    v[2]  = time % 24
    v[3]  = amount * 0.01
    v[4]  = np.sin(time)
    v[5]  = np.cos(time)
    v[6]  = np.sqrt(amount) if amount > 0 else 0
    v[7]  = amount / (time + 1)
    v[8]  = (time % 60) / 60
    v[9]  = amount * np.sin(time)
    v[10] = np.log1p(time)
    v[11] = amount ** 0.5
    v[12] = amount / (v[11] + 1)
    v[13] = np.tanh(amount)
    v[14] = np.tanh(time)
    v[15] = amount * 0.001
    v[16] = time * 0.001
    v[17] = np.sin(amount)
    v[18] = np.cos(amount)
    v[19] = amount / 500
    v[20] = time / 100000
    v[21] = np.log1p(amount + time)
    v[22] = (amount % 100) / 100
    v[23] = (time % 1000) / 1000
    v[24] = amount * time * 1e-6
    v[25] = np.exp(-amount / 10000)
    v[26] = np.exp(-time / 100000)
    v[27] = (amount - time) * 1e-5

    return v.tolist()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    fraud_percent = None
    risk = None

    if request.method == "POST":
        # Read user inputs
        time = float(request.form["time"])
        amount = float(request.form["amount"])

        # Threshold from UI (default = 0.5)
        threshold = float(request.form.get("threshold", 0.5))

        # Generate backend V features
        v_features = generate_v_features(amount, time)

        # Prepare input data
        data = np.array([[time] + v_features + [amount]])
        data_scaled = scaler.transform(data)

        # ---- PART 1: Fraud Probability ----
        probability = model.predict_proba(data_scaled)[0][1]
        fraud_percent = round(probability * 100, 2)

        # ---- PART 2: Threshold Control ----
        if probability >= threshold:
            result = "Fraudulent Transaction"
        else:
            result = "Legitimate Transaction"

        # Risk Level
        if fraud_percent < 30:
            risk = "LOW"
        elif fraud_percent < 70:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # Save for dashboard
        prediction_log.append({
            "Probability (%)": fraud_percent,
            "Risk": risk,
            "Result": result
        })

        prediction_log[:] = prediction_log[-20:]

    return render_template(
        "index.html",
        result=result,
        fraud_percent=fraud_percent,
        risk=risk
    )


if __name__ == "__main__":
    app.run(debug=False)
