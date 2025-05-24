import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

# Ensure folder exists for saving model
os.makedirs("trained_data", exist_ok=True)

# Load dataset
df = pd.read_csv("csv/Students_Grading_Dataset.csv")

# Create 'Pass' column (1 = Pass, 0 = Fail)
df['Pass'] = (df['Total_Score'] >= 60).astype(int)

# Features and target
X = df[['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Participation_Score', 'Attendance (%)']]
y = df['Pass']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with scaler and MLPClassifier (sigmoid used internally)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=1000, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'trained_data/student_pass_fail_model.pkl')

print("Model trained and saved successfully.")
