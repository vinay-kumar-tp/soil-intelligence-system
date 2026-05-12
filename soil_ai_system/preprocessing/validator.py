def validate_input(row: dict) -> dict:
    """Validate a single input record against range constraints.

    Args:
        row (dict): Raw input values.

    Returns:
        dict: Validation result with "valid" flag and list of errors.
    """
    errors = []

    if not (0 <= row.get("ph", -1) <= 14):
        errors.append("pH must be between 0 and 14")
    if not (0 <= row.get("N", -1) <= 200):
        errors.append("Nitrogen must be 0-200 kg/ha")
    if not (0 <= row.get("P", -1) <= 200):
        errors.append("Phosphorus must be 0-200 kg/ha")
    if not (0 <= row.get("K", -1) <= 200):
        errors.append("Potassium must be 0-200 kg/ha")
    if not (0 <= row.get("moisture", -1) <= 100):
        errors.append("Moisture must be 0-100 percent")
    if not (0 <= row.get("humidity", -1) <= 100):
        errors.append("Humidity must be 0-100 percent")
    if not (0 <= row.get("rainfall", -1) <= 5000):
        errors.append("Rainfall must be 0-5000 mm")
    if row.get("ec", 0) > 4.0:
        errors.append("EC above 4.0 indicates extreme salinity - verify reading")
    if row.get("season") not in ["kharif", "rabi", "summer", None]:
        errors.append("Season must be kharif, rabi, or summer")

    return {"valid": len(errors) == 0, "errors": errors}


def validate_dataframe(df):
    """Validate dataset-level constraints before training.

    Args:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        dict: Summary of validation checks and dataset stats.
    """
    report = {}
    report["shape"] = df.shape
    report["nulls"] = df.isnull().sum().to_dict()
    report["duplicates"] = df.duplicated().sum()
    report["ph_range_ok"] = df["ph"].between(0, 14).all()
    report["N_range_ok"] = df["N"].between(0, 200).all()
    return report
