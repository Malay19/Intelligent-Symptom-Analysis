import csv
import pandas as pd

def get_specific_line(filename, line_number):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if i == line_number:
                return row
    return None

# Get line 8134 (originally from sed command)
line_8134 = get_specific_line('Final_Augmented_dataset_Diseases_and_Symptoms.csv', 8134)
print(f"Line 8134 contents: {line_8134}")
with open('Final_Augmented_dataset_Diseases_and_Symptoms.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

# Find max columns
max_cols = max(len(row) for row in rows)

# Pad rows with fewer columns
cleaned_rows = [row + ['']*(max_cols-len(row)) for row in rows]

# Rewrite cleaned file
with open('Cleaned_Dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(cleaned_rows)

import csv
import pandas as pd

# Read the original file
with open('Final_Augmented_dataset_Diseases_and_Symptoms.csv', 'r', encoding='utf-8-sig') as f:
    # Peek at the first few lines to identify structure
    first_lines = [next(f) for _ in range(5)]
    print("First 5 lines of the file:")
    for line in first_lines:
        print(line[:100])  # Print first 100 chars of each line

    # Reset file pointer
    f.seek(0)

    # Read the file
    reader = csv.reader(f)
    rows = [row for row in reader]

# Find max columns and print information for debugging
max_cols = max(len(row) for row in rows)
print(f"Found {len(rows)} rows with max {max_cols} columns")
print(f"Sample first row column count: {len(rows[0])}")
print(f"First few values in first row: {rows[0][:5]}")

# Check for potential disease column
if len(rows) > 1:  # Make sure we have at least a header and one data row
    # Get the first column values from first few rows
    first_col_values = [row[0] if row else "EMPTY" for row in rows[:10]]
    print(f"First column values: {first_col_values}")

    # Check if first column might have disease names
    unique_first_col = set(row[0] for row in rows if row)
    print(f"First column has {len(unique_first_col)} unique values")
    print(f"Sample unique values: {list(unique_first_col)[:5]}")

# Pad rows with fewer columns
cleaned_rows = [row + ['']*(max_cols-len(row)) for row in rows]

# Create a DataFrame and examine
df = pd.DataFrame(cleaned_rows)
# If first row looks like headers, use it as column names
if any(str(val).lower() in ['disease', 'diseases', 'condition'] for val in df.iloc[0]):
    df.columns = df.iloc[0]
    df = df[1:]  # Remove header row
    print("Using first row as headers")
else:
    print("First row doesn't appear to contain headers")
    # Explicitly name first column as 'disease'
    column_names = ['disease'] + [f'symptom_{i}' for i in range(1, len(df.columns))]
    df.columns = column_names
    print(f"Renamed columns: {column_names[:5]}...")

# Save cleaned DataFrame
df.to_csv('Better_Cleaned_Dataset.csv', index=False)
print(f"Saved cleaned dataset with shape: {df.shape}")
print(f"Column names: {df.columns[:5]}...")
print(f"First 3 rows of data:")
print(df.head(3))