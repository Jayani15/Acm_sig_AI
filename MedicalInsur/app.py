import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Medical Insurance - Minimal E2E", layout="centered")

@st.cache_data
def load_data(path="MedicalInsurance.csv"):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def build_and_train(df):
    X = df.drop(columns=["charges"])
    y = df["charges"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ["sex", "smoker", "region"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("pre", preproc),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return model, mean_absolute_error(y_val, preds), r2_score(y_val, preds)

def predict_ui(model):
    st.header("Single prediction")
    # Basic input widgets
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)
    children = st.slider("Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", sorted(["southwest","southeast","northwest","northeast"]))

    if st.button("Predict"):
        X_new = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        pred = model.predict(X_new)[0]
        st.metric("Predicted charges (USD)", f"{pred:,.2f}")

def main():
    st.title("Minimal Medical Insurance — Streamlit E2E")
    st.write("Load data → train model → make predictions. This trains in-session (no file saves).")

    df = load_data()

    page = st.sidebar.selectbox("Page", ["Data", "Train & Evaluate", "Predict"])

    if page == "Data":
        st.header("Dataset preview")
        st.dataframe(df.head(100))
        st.write("Summary")
        st.write(df.describe(include="all"))

    elif page == "Train & Evaluate":
        st.header("Train model")
        if st.button("Train model now"):
            with st.spinner("Training..."):
                model, mae, r2 = build_and_train(df)
                st.success("Training finished")
                st.write(f"MAE: {mae:,.2f}")
                st.write(f"R²: {r2:.3f}")
                st.session_state["model"] = model
        else:
            st.info("Click 'Train model now' to train a RandomForest on the dataset in memory.")

    elif page == "Predict":
        if "model" not in st.session_state:
            st.info("Model not found in session — training automatically (quick).")
            with st.spinner("Training..."):
                model, mae, r2 = build_and_train(df)
                st.session_state["model"] = model
                st.success(f"Trained (MAE {mae:,.2f}, R² {r2:.3f})")
        predict_ui(st.session_state["model"])

if __name__ == "__main__":
    main()