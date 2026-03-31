# ==============================
# DIGITAL TWIN - COVID PROJECT
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------------
# STEP 1: LOAD DATA
# ------------------------------
data = pd.read_csv("covid_india.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ------------------------------
# STEP 2: PREPROCESS DATA
# ------------------------------
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Keep only required columns
data = data[['Date', 'Confirmed']]

# Create lag feature (previous day cases)
data['Prev_Day'] = data['Confirmed'].shift(1)

# Remove null values
data = data.dropna()

print("\nProcessed Data:")
print(data.head())

# ------------------------------
# STEP 3: TRAIN MODEL
# ------------------------------
X = data[['Prev_Day']]
y = data['Confirmed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ------------------------------
# STEP 4: PREDICTION
# ------------------------------
predictions = model.predict(X_test)

print("\nSample Predictions:")
print(predictions[:5])

# ------------------------------
# STEP 5: DIGITAL TWIN SIMULATION
# ------------------------------
print("\n--- DIGITAL TWIN SIMULATION ---")

# Example: Input today's cases
today_cases = float(input("Enter today's COVID cases: "))

future_prediction = model.predict([[today_cases]])
predicted_cases = future_prediction[0]

print(f"\nPredicted Tomorrow Cases: {predicted_cases:.2f}")

# ------------------------------
# STEP 6: RESOURCE ESTIMATION
# ------------------------------
hospital_rate = 0.2
oxygen_rate = 0.05

beds_needed = predicted_cases * hospital_rate
oxygen_needed = predicted_cases * oxygen_rate

print("\n--- RESOURCE REQUIREMENT ---")
print(f"Beds Needed: {beds_needed:.0f}")
print(f"Oxygen Needed: {oxygen_needed:.0f}")

# ------------------------------
# STEP 7: VISUALIZATION
# ------------------------------
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Confirmed'], label="Actual Cases")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("COVID-19 Trend (Digital Twin View)")
plt.legend()
plt.show()