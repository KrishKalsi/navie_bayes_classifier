import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score


st.title("📊 Naive Bayes Classifier App")

st.write("Upload a dataset and train a Naive Bayes model.")

# --------------------------
# File Upload
# --------------------------

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------
    # Target Selection
    # --------------------------

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # --------------------------
    # Train Test Split
    # --------------------------

    test_size = st.slider(
        "Test Size (Train-Test Split)",
        0.1, 0.5, 0.2
    )

    # --------------------------
    # Model Selection
    # --------------------------

    model_name = st.selectbox(
        "Choose Naive Bayes Model",
        ["GaussianNB", "MultinomialNB", "BernoulliNB"]
    )

    # --------------------------
    # Train Button
    # --------------------------

    if st.button("Train Model"):

        try:

            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Encode categorical variables
            X = pd.get_dummies(X)

            # Encode target if categorical
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )

            # Model selection
            if model_name == "GaussianNB":
                model = GaussianNB()

            elif model_name == "MultinomialNB":
                model = MultinomialNB()

            else:
                model = BernoulliNB()

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Accuracy
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Output
            st.subheader("Results")

            st.write(f"✅ Training Accuracy: **{train_acc:.4f}**")
            st.write(f"✅ Testing Accuracy: **{test_acc:.4f}**")

        except Exception as e:
            st.error(f"Error: {e}")