import csv
from typing import Any, Dict


def profile_dataset(file_path: str) -> Dict[str, Any]:
	"""Return a lightweight profile for tabular datasets.

	Supports CSV with Python stdlib so profiling works without optional data packages.
	"""
	with open(file_path, "r", newline="", encoding="utf-8") as fp:
		reader = csv.reader(fp)
		rows = list(reader)

	if not rows:
		return {
			"row_count": 0,
			"column_count": 0,
			"columns": [],
			"missing_values": {},
			"dtypes": {},
		}

	header = rows[0]
	data_rows = rows[1:]
	col_count = len(header)

	missing_values = {column: 0 for column in header}
	for row in data_rows:
		for idx, column in enumerate(header):
			value = row[idx] if idx < len(row) else ""
			if value is None or str(value).strip() == "":
				missing_values[column] += 1

	return {
		"row_count": len(data_rows),
		"column_count": col_count,
		"columns": header,
		"missing_values": missing_values,
		"dtypes": {column: "unknown" for column in header},
	}
