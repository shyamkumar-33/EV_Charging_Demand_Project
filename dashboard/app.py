
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Find the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "cleaned_charging_data.csv"
REGRESSION_PATH = PROJECT_DIR / "models" / "best_regression_model.joblib"
CLASSIFICATION_PATH = PROJECT_DIR / "models" / "best_classification_model.joblib"
FORECAST_PATH = PROJECT_DIR / "models" / "seven_day_forecast.csv"

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EV Charging Demand Prediction")
st.write(
    "Explore historical charging records, predict demand, "
    "and review the saved seven-day forecast."
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load historical data
try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the cleaned dataset: {exc}")
    st.stop()

# Overview
st.header("1. Historical Data Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Charging records", f"{len(df):,}")
col2.metric("Dataset columns", len(df.columns))

if "Demand" in df.columns:
    col3.metric(
        "Average recorded demand",
        f"{df['Demand'].mean():.2f}"
        if df["Demand"].notna().any()
        else "N/A",
    )
else:
    col3.metric("Demand column", "Not found")

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# Historical demand chart
if "StartDatetime" in df.columns and "Demand" in df.columns:
    st.subheader("Historical Demand")

    chart_df = df[["StartDatetime", "Demand"]].copy()
    chart_df["StartDatetime"] = pd.to_datetime(
        chart_df["StartDatetime"], errors="coerce"
    )
    chart_df["Demand"] = pd.to_numeric(
        chart_df["Demand"], errors="coerce"
    )

    chart_df = chart_df.dropna().sort_values("StartDatetime")

    if not chart_df.empty:
        chart_df = chart_df.set_index("StartDatetime")
        daily = chart_df["Demand"].resample("D").sum(min_count=1)
        st.line_chart(daily.rename("Daily recorded demand"))
    else:
        st.info("No valid date and demand values are available for this chart.")

# Regression prediction form
st.header("2. Predict Charging Demand")
st.caption(
    "Enter the same input fields used when training the regression model. "
    "The app will report the model output, not a guaranteed real-world value."
)

try:
    regression_model = load_model(REGRESSION_PATH)

    # Use the model's original input columns where available
    if hasattr(regression_model, "feature_names_in_"):
        feature_columns = list(regression_model.feature_names_in_)
    elif hasattr(regression_model, "named_steps"):
        preprocessor = regression_model.named_steps.get("preprocessor")
        feature_columns = (
            list(preprocessor.feature_names_in_)
            if preprocessor is not None
            and hasattr(preprocessor, "feature_names_in_")
            else []
        )
    else:
        feature_columns = []

    if not feature_columns:
        st.warning(
            "Could not automatically determine the model's input columns. "
            "Check the training notebook to confirm them."
        )
    else:
        input_values = {}

        with st.form("prediction_form"):
            for column in feature_columns:
                if column not in df.columns:
                    st.warning(
                        f"Training input '{column}' was not found in the dataset."
                    )
                    continue

                if pd.api.types.is_numeric_dtype(df[column]):
                    median = pd.to_numeric(
                        df[column], errors="coerce"
                    ).median()
                    default = float(median) if pd.notna(median) else 0.0

                    input_values[column] = st.number_input(
                        column,
                        value=default,
                    )
                else:
                    choices = (
                        df[column]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .index
                        .tolist()
                    )

                    if choices:
                        input_values[column] = st.selectbox(
                            column, choices
                        )
                    else:
                        input_values[column] = st.text_input(column)

            submitted = st.form_submit_button("Predict demand")

        if submitted and len(input_values) == len(feature_columns):
            input_df = pd.DataFrame(
                [input_values], columns=feature_columns
            )

            try:
                prediction = regression_model.predict(input_df)[0]
                st.success(f"Predicted demand: {prediction:.2f}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

except FileNotFoundError:
    st.error("Regression model not found. Run the regression notebook first.")
except Exception as exc:
    st.error(f"Could not load the regression model: {exc}")

# Forecast results
st.header("3. Seven-Day Forecast")

if FORECAST_PATH.exists():
    try:
        forecast_df = pd.read_csv(FORECAST_PATH)
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button(
            "Download seven-day forecast",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="seven_day_forecast.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Could not read forecast file: {exc}")
else:
    st.info("Run notebook 04_time_series_forecasting.ipynb to create the forecast.")

st.caption("EV Charging Demand Prediction — Academic Project")
