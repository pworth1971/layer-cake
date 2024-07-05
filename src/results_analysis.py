import pandas as pd
import sys
from tabulate import tabulate


# ----------------------------------------------------------------------------------------------------------------------------
# results_analysis()
#
# analyze the model performance results, print summary either to sdout or file
#
# ----------------------------------------------------------------------------------------------------------------------------

def results_analysis(file_path, output_path=None):
    
    df = pd.read_csv(file_path, sep='\t')           # Load data from a CSV file (tab delimited)
    
    print("Columns in the file:", df.columns)

    # Group data by 'dataset', 'embedding', 'model', 'wc-supervised' and get the maximum 'value'
    #result = df.groupby(['dataset', 'embeddings', 'model', 'wc-supervised', 'measure'])['value'].max().reset_index()
    result = df.groupby(['dataset', 'embeddings', 'model', 'wc-supervised', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df, result, how='inner', on=['dataset', 'embeddings', 'model', 'wc-supervised', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'embeddings', 'model', 'wc-supervised', 'measure', 'value'])

    # Specify the column order
    columns_order = ['dataset', 'model', 'pretrained', 'embeddings', 'wc-supervised', 'measure',
                     'params', 'tunable', 'value', 'run', 'epoch']
    final_result = unique_result[columns_order]
    
    # Sorting results for better readability
    #result = result.sort_values(by=['dataset', 'embeddings', 'model', 'wc-supervised', 'measure'])
    
    if output_path:                                 # Output to a file
        with open(output_path, 'w') as f:
            f.write(tabulate(final_result, headers='keys', tablefmt='pretty', showindex=False))
        print(f"Output saved to {output_path}")
    else:                                           # output to stdout
        print(tabulate(final_result, headers='keys', tablefmt='pretty', showindex=False))
        #print(result.to_string(index=False))            # Print to standard output

# ----------------------------------------------------------------------------------------------------------------------------
#     

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_data.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    results_analysis(input_file, output_file)

if __name__ == "__main__":
    main()
