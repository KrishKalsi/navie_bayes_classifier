import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Classification Models
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score


st.title("📊 Machine Learning Model Builder")

st.write("Upload dataset, select features, choose task, and train model.")

# ========================
# File Upload
# ========================

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # ========================
    # Feature Selection
    # ========================

    X_columns = st.multiselect(
        "Select Feature Columns (X)",
        columns
    )

    y_column = st.selectbox(
        "Select Target Column (y)",
        columns
    )

    # ========================
    # Task Selection
    # ========================

    task_type = st.radio(
        "Select Task Type",
        ["Classification", "Regression"]
    )

    # ========================
    # Train Test Split
    # ========================

    test_size = st.slider(
        "Test Size",
        0.1, 0.5, 0.2
    )

    # ========================
    # Model Selection
    # ========================

    if task_type == "Classification":
        model_name = st.selectbox(
            "Choose Model",
            ["GaussianNB", "MultinomialNB", "BernoulliNB"]
        )
    else:
        model_name = st.selectbox(
            "Choose Model",
            ["Linear Regression"]
        )

    # ========================
    # Train Button
    # ========================

    if st.button("Train Model"):

        try:

            if len(X_columns) == 0:
                st.error("Please select at least one feature column.")
                st.stop()

            if y_column in X_columns:
                st.error("Target column cannot be in feature columns.")
                st.stop()

            X = df[X_columns]
            y = df[y_column]

            # Encode categorical features
            X = pd.get_dummies(X)

            # ========================
            # TASK VALIDATION
            # ========================

            if task_type == "Classification":

                # Check if classification possible
                if y.nunique() > 20:
                    st.error(
                        "❌ Classification may not be suitable: Target has too many unique values."
                    )
                    st.stop()

                if y.dtype == "object":
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                # Model selection
                if model_name == "GaussianNB":
                    model = GaussianNB()

                elif model_name == "MultinomialNB":
                    if (X < 0).any().any():
                        st.error(
                            "❌ MultinomialNB requires non-negative features."
                        )
                        st.stop()
                    model = MultinomialNB()

                else:
                    model = BernoulliNB()

            else:  # Regression

                # Check if regression possible
                if not np.issubdtype(y.dtype, np.number):
                    st.error(
                        "❌ Regression not possible: Target must be numeric."
                    )
                    st.stop()

                model = LinearRegression()

            # ========================
            # Train Test Split
            # ========================

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # ========================
            # Metrics
            # ========================

            st.subheader("Results")

            if task_type == "Classification":

                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)

                st.write(f"✅ Training Accuracy: **{train_acc:.4f}**")
                st.write(f"✅ Testing Accuracy: **{test_acc:.4f}**")

            else:

                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                st.write(f"✅ Training R² Score: **{train_r2:.4f}**")
                st.write(f"✅ Testing R² Score: **{test_r2:.4f}**")

        except Exception as e:
            st.error(f"Error: {e}")