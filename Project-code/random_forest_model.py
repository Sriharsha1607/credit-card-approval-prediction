# ==========================
# RANDOM FOREST CLASSIFICATION
# Credit Card Approval Prediction
# ==========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# Load dataset
df = pd.read_csv("Application_Data.csv")


# --------------------------
# Data Preprocessing
# --------------------------

# Remove Applicant_ID (not useful for prediction)
if 'Applicant_ID' in df.columns:
    df.drop('Applicant_ID', axis=1, inplace=True)

# Label Encoding for categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))


# --------------------------
# Feature Selection
# --------------------------

X = df.drop('Status', axis=1)   # Features
y = df['Status']                # Target


# --------------------------
# Train-Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------
# Random Forest Function
# --------------------------

def random_forest(X_train, X_test, y_train, y_test):
    """
    Builds, trains, and tests a Random Forest Classification model,
    returning performance metrics.
    """

    # Initialize Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)

    # Generate predictions
    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)

    # Evaluation
    print("\n" + "=" * 40)
    print("Random Forest Model Evaluation")
    print("=" * 40)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return rf_model


# --------------------------
# Run Random Forest
# --------------------------

rf_model = random_forest(X_train, X_test, y_train, y_test)
