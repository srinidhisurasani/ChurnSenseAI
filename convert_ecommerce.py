import json
import pandas as pd

input_file = "data/uploads/E Commerce Dataset.xlsx"
output_file = "data/uploads/ecommerce_dataset.csv"

print("Reading E-Commerce dataset...")

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

print("JSON file loaded successfully.")

cells = data.get("cells", [])

rows = []

for cell in cells:
    rows.append(cell)

df = pd.DataFrame(rows)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

df.to_csv(output_file, index=False)

print("\nCSV file created successfully:")
print(output_file)