import pandas as pd
import os

# Construct the correct file path
file_path = os.path.join("..", "log", "test", "ml_def_full_test.test")

# Check if the file exists before reading
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the tab-delimited file
df = pd.read_csv(file_path, delimiter='\t')

# Iterate through rows and modify the embeddings column
for index, row in df.iterrows():
    if row['embeddings'].endswith('vmode'):
        print(f"Found vmode at row {index}: {row['embeddings']}")

        last_comp_method = row['comp_method'].split(':')[-1]  # Extract last component
        print("Last component of comp_method:", last_comp_method)

        # Append to embeddings column
        df.at[index, 'embeddings'] += '.' + last_comp_method

        print("Updated embeddings:", df.at[index, 'embeddings'])

# Construct the output file path
output_file = os.path.join("..", "log", "test", "ml_def_full_test.test.updated")
df.to_csv(output_file, sep='\t', index=False)

print(f"Updated file saved as {output_file}")

