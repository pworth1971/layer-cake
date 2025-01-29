import pandas as pd
import argparse
import os

from model.LCRepresentationModel import MODEL_MAP, MODEL_DIR
from util.common import get_model_identifier

VECTOR_CACHE = "../.vector_cache"  # Define the cache directory

def update_tsv(input_file, output_file="output_modified.tsv", neural=True):
    """
    Reads a TSV file, extracts the model name from the 'embeddings' column,
    adds a 'model' column using get_model_identifier, sorts columns alphabetically,
    and saves to a new TSV file.

    Args:
        input_file (str): Path to the input TSV file.
        output_file (str): Path to the output TSV file (default: output_modified.tsv).
        neural (bool): Whether to extract the full embedding model name (True) or just the first part before the colon (False).
    """

    print(f"updating log file {input_file}, output file: {output_file}, neural: {neural}")

    # Load the TSV file
    df = pd.read_csv(input_file, sep="\t")

    print("Columns in the TSV file:")
    print(df.columns.tolist())
    
    #print("First few rows of the TSV file:")
    #print(df.head())
    
    # Function to extract the model name
    def extract_model_from_embeddings(embedding_value):

        if pd.isna(embedding_value):
            return None  # Handle missing values
        
        if not neural:
            #print("neural is false, extracting model from embeddings column...")
            embedding_value = embedding_value.split(":")[0]  # Take only the first part before the colon
            #print("embedding_value: ", embedding_value)

        model_name, _ = get_model_identifier(embedding_value.lower())  # Ensure case-insensitivity

        #print("model_name: ", model_name)
        return model_name

    # add 'classifier' column from 'model' column 
    df["classifier"] = df["model"]
    
    # change 'model' column to the embedding model name
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
    parser.add_argument("--neural", action="store_true", default=False, help="Extract full embedding model name (default: False, only takes first part before ':').")

    args = parser.parse_args()
    update_tsv(args.input_file, args.output_file, args.neural)

