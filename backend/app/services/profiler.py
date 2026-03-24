import csv
from typing import Any, Dict

def profile_dataset(file_path: str) -> Dict[str, Any]:
    """Return a profile for tabular datasets with basic type inference."""
    with open(file_path, "r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.reader(fp)
        rows = list(reader)

    if not rows:
        return {
            "row_count": 0, "column_count": 0, "columns": [],
            "missing_values": {}, "dtypes": {},
        }

    # 1. Clean Headers (handles literal and invisible BOMs)
    header = [str(column).replace("\ufeff", "").replace("\\ufeff", "").strip() for column in rows[0]]
    data_rows = rows[1:]
    col_count = len(header)
    row_count = len(data_rows)

    # 2. Initialize profiling containers
    missing_values = {column: 0 for column in header}
    # We start by assuming everything is an integer, then "demote" as we find complexity
    inferred_types = {column: "int" for column in header}

    # 3. Process Rows
    for row in data_rows:
        for idx, col_name in enumerate(header):
            # Guard against short rows
            val = row[idx].strip() if idx < len(row) else ""
            
            # Track missing values
            if val == "":
                missing_values[col_name] += 1
                continue

            # Type Inference Logic
            current_type = inferred_types[col_name]
            if current_type == "string":
                continue  # Already at the lowest common denominator

            try:
                if current_type == "int":
                    int(val)
                elif current_type == "float":
                    float(val)
            except ValueError:
                # Promotion logic: int -> float -> string
                if current_type == "int":
                    try:
                        float(val)
                        inferred_types[col_name] = "float"
                    except ValueError:
                        inferred_types[col_name] = "string"
                else:
                    inferred_types[col_name] = "string"

    return {
        "row_count": row_count,
        "column_count": col_count,
        "columns": header,
        "missing_values": missing_values,
        "dtypes": inferred_types,
    }