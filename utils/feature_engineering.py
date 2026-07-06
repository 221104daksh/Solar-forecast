"""Converts raw live weather data into the exact feature dataframe the
trained Random Forest model expects.

IMPORTANT: the model was trained with features in this EXACT order:
    1. IRRADIATION
    2. Hour
    3. MODULE_TEMPERATURE
    4. AMBIENT_TEMPERATURE
This order must never change.
"""

from __future__ import annotations

import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = ["IRRADIATION", "Hour", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]

# Open-Meteo's shortwave_radiation is in W/m^2. Solar plant datasets (the kind
# these Random Forest models are typically trained on) commonly express
# IRRADIATION as a fraction of the ~1000 W/m^2 "standard test condition".
# Adjust this constant if your training data used a different scale.
RADIATION_TO_IRRADIATION_DIVISOR = 1000.0
IRRADIATION_CLIP_MIN, IRRADIATION_CLIP_MAX = 0.0, 1.2

# Realistic module temperature bounds for clipping the heuristic estimate.
MODULE_TEMP_CLIP_MIN, MODULE_TEMP_CLIP_MAX = -10.0, 85.0


def shortwave_radiation_to_irradiation(shortwave_radiation: float) -> float:
    """Converts Open-Meteo shortwave radiation (W/m^2) into the model's
    IRRADIATION feature scale, clipped to a realistic range.
    """
    irradiation = shortwave_radiation / RADIATION_TO_IRRADIATION_DIVISOR
    return float(min(max(irradiation, IRRADIATION_CLIP_MIN), IRRADIATION_CLIP_MAX))


def estimate_module_temperature(ambient_temperature: float, shortwave_radiation: float) -> float:
    """Estimates solar module temperature since Open-Meteo does not provide it.

    Formula (per project spec):
        Module Temperature = Ambient Temperature + (Shortwave Radiation / 800) * 20
    """
    estimated = ambient_temperature + (shortwave_radiation / 800.0) * 20.0
    clipped = min(max(estimated, MODULE_TEMP_CLIP_MIN), MODULE_TEMP_CLIP_MAX)
    if clipped != estimated:
        logger.info("Module temperature %.2f clipped to %.2f", estimated, clipped)
    return float(clipped)


def build_feature_dataframe(
    irradiation: float,
    hour: int,
    module_temperature: float,
    ambient_temperature: float,
) -> pd.DataFrame:
    """Builds the single-row dataframe in the exact column order the model expects."""
    return pd.DataFrame(
        [[irradiation, hour, module_temperature, ambient_temperature]],
        columns=FEATURE_COLUMNS,
    )


def engineer_features_from_weather(
    ambient_temperature: float,
    shortwave_radiation: float,
    hour: int,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: raw weather -> engineered features -> model-ready dataframe.

    Returns the feature dataframe plus a dict of the intermediate values
    (irradiation, module_temperature) for display in the UI.
    """
    irradiation = shortwave_radiation_to_irradiation(shortwave_radiation)
    module_temperature = estimate_module_temperature(ambient_temperature, shortwave_radiation)

    features_df = build_feature_dataframe(
        irradiation=irradiation,
        hour=hour,
        module_temperature=module_temperature,
        ambient_temperature=ambient_temperature,
    )

    intermediate = {
        "irradiation": irradiation,
        "module_temperature": module_temperature,
        "hour": hour,
    }
    return features_df, intermediate
