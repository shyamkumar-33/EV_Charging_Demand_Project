
from pathlib import Path

import pandas as pd

from model_utils import (
    load_regression_model,
    load_classification_model,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "cleaned_charging_data.csv"


def get_input_columns(model):
    """Get the original columns expected by the model."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")

        if preprocessor is not None and hasattr(
            preprocessor, "feature_names_in_"
        ):
            return list(preprocessor.feature_names_in_)

    raise ValueError(
        "Could not identify model input columns. "
        "Check the model training notebook."
    )


def main():
    df = pd.read_csv(DATA_PATH)

    regression_model = load_regression_model()
    classification_model = load_classification_model()

    regression_columns = get_input_columns(regression_model)
    classification_columns = get_input_columns(classification_model)

    # Ensure the dataset has the columns needed by both models
    required_columns = set(regression_columns + classification_columns)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Use a real record as a smoke test
    sample = df.dropna(
        subset=list(required_columns)
    ).head(1)

    if sample.empty:
        raise ValueError("No complete sample record is available.")

    regression_input = sample[regression_columns]
    classification_input = sample[classification_columns]

    regression_prediction = regression_model.predict(regression_input)[0]
    classification_prediction = classification_model.predict(
        classification_input
    )[0]

    print("Both trained models loaded successfully!")
    print("Sample record index:", sample.index[0])
    print(f"Predicted charging demand: {regression_prediction:.2f}")
    print(f"Predicted demand class: {classification_prediction}")


if __name__ == "__main__":
    main()
