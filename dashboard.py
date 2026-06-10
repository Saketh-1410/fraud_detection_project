import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Fraud Detection Dashboard")

# Sample statistics (demo purpose)
total_transactions = 1000
fraud_count = 32
legit_count = total_transactions - fraud_count

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_transactions)
col2.metric("Fraud Detected", fraud_count)
col3.metric("Legitimate", legit_count)

st.markdown("---")

# Bar Chart
data = pd.DataFrame({
    "Type": ["Legitimate", "Fraud"],
    "Count": [legit_count, fraud_count]
})

st.subheader("Fraud vs Legitimate Transactions")

fig, ax = plt.subplots()
ax.bar(data["Type"], data["Count"])
st.pyplot(fig)

st.markdown("---")

st.subheader("Key Insights")
st.write("""
- Fraud cases are rare compared to legitimate transactions  
- Model prioritizes reducing false positives  
- Dashboard helps monitor fraud trends in real time  
""")
