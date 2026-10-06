
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_insurance_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

except Exception as e:
    st.error("Unable to load the saved model.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# GET PREPROCESSOR
# --------------------------------------------------

try:
    preprocessor = model.named_steps["preprocessor"]

except Exception:
    st.error(
        "The saved model does not contain a step named 'preprocessor'. "
        "Please check how the model was saved."
    )
    st.stop()

# --------------------------------------------------
# FIND NUMERIC AND CATEGORICAL COLUMNS
# --------------------------------------------------

numeric_columns = []
categorical_columns = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "remainder":
        continue

    if name.lower() in ["num", "numeric"]:
        numeric_columns = list(columns)

    elif name.lower() in ["cat", "categorical"]:
        categorical_columns = list(columns)

# --------------------------------------------------
# GET CATEGORICAL OPTIONS FROM TRAINED ENCODER
# --------------------------------------------------

categorical_options = {}

for name, transformer, columns in preprocessor.transformers_:

    if name.lower() in ["cat", "categorical"]:

        # If categorical transformer is itself a Pipeline
        if hasattr(transformer, "named_steps"):

            encoder = None

            for step_name, step in transformer.named_steps.items():

                if hasattr(step, "categories_"):
                    encoder = step
                    break

        else:
            encoder = transformer

        if encoder is not None and hasattr(encoder, "categories_"):

            for column, categories in zip(
                columns,
                encoder.categories_
            ):
                categorical_options[column] = [
                    str(x) for x in categories
                ]

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏥 Medical Insurance Cost Prediction")

st.write(
    "Enter patient information to predict the annual medical insurance cost."
)

st.divider()

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("Model Information"):

    st.write("**Model:**", type(model.named_steps["model"]).__name__)

    st.write("**Numeric Features:**")
    st.write(numeric_columns)

    st.write("**Categorical Features:**")
    st.write(categorical_columns)

# --------------------------------------------------
# INPUT DATA
# --------------------------------------------------

st.subheader("👤 Patient Information")

input_data = {}

# --------------------------------------------------
# NUMERIC INPUTS
# --------------------------------------------------

if numeric_columns:

    st.write("### Numerical Information")

    numeric_cols = st.columns(3)

    for i, column in enumerate(numeric_columns):

        with numeric_cols[i % 3]:

            input_data[column] = st.number_input(
                column.replace("_", " ").title(),
                value=0.0
            )

# --------------------------------------------------
# CATEGORICAL INPUTS
# --------------------------------------------------

if categorical_columns:

    st.write("### Categorical Information")

    categorical_cols = st.columns(3)

    for i, column in enumerate(categorical_columns):

        with categorical_cols[i % 3]:

            options = categorical_options.get(
                column,
                ["Unknown"]
            )

            input_data[column] = st.selectbox(
                column.replace("_", " ").title(),
                options
            )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Medical Insurance Cost",
    use_container_width=True
):

    try:

        # Create dataframe using EXACT training columns
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Make prediction
        prediction = model.predict(input_df)[0]

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.success("Prediction generated successfully!")

        st.metric(
            "Predicted Annual Medical Cost",
            f"₹ {prediction:,.2f}"
        )

        st.subheader("📋 Patient Input")

        st.dataframe(
            input_df,
            use_container_width=True
        )

    except Exception as e:

        st.error("Prediction could not be generated.")

        st.exception(e)

