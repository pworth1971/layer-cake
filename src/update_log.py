import pandas as pd
import argparse
import os

from model.LCRepresentationModel import MODEL_MAP, MODEL_DIR


VECTOR_CACHE = "../.vector_cache"  # Define the cache directory

# Function to get model name and path
def get_model_identifier(pretrained, cache_dir=VECTOR_CACHE):
    """
    Get the full model identifier based on pretrained model name.

    Args:
        pretrained (str): Model name from the embeddings column.
        cache_dir (str): Directory to load model from.

    Returns:
        tuple: (model_name, model_path)
    """
    model_name = MODEL_MAP.get(pretrained, pretrained)
    model_dir = MODEL_DIR.get(pretrained, pretrained)
    model_path = os.path.join(cache_dir, model_dir)
    return model_name, model_path



def update_tsv(input_file, output_file="output_modified.tsv"):
    """
    Reads a TSV file, extracts the model name from the 'embeddings' column,
    adds a 'model' column using get_model_identifier, sorts columns alphabetically,
    and saves to a new TSV file.

    Args:
        input_file (str): Path to the input TSV file.
        output_file (str): Path to the output TSV file (default: output_modified.tsv).
    """
    # Load the TSV file
    df = pd.read_csv(input_file, sep="\t")

    print("Columns in the TSV file:")
    print(df.columns.tolist())
    
    print("First few rows of the TSV file:")
    print(df.head())
    
    # Create the "model" column using get_model_identifier
    def extract_model_from_embeddings(embedding_value):
        if pd.isna(embedding_value):
            return None  # Handle missing values
        model_name, _ = get_model_identifier(embedding_value.lower())  # Ensure case-insensitivity
        return model_name

    df["model"] = df["embeddings"].apply(extract_model_from_embeddings)
    print("First few rows after adding 'model' column:")
    print(df.head())
    
    # Sort columns alphabetically
    df = df[sorted(df.columns)]

    # Save the modified TSV file
    df.to_csv(output_file, sep="\t", index=False)

    print(f"TSV file processed successfully. Saved as '{output_file}'.")

# Command-line argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a TSV file, add 'model' column, and sort columns alphabetically.")
    parser.add_argument("input_file", help="Path to the input TSV file.")
    parser.add_argument("output_file", nargs="?", default="output_modified.tsv", help="Path to the output TSV file (default: output_modified.tsv)")

    args = parser.parse_args()
    update_tsv(args.input_file, args.output_file)


