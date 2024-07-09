import pandas as pd
import sys
from tabulate import tabulate


# ----------------------------------------------------------------------------------------------------------------------------
# results_analysis()
#
# analyze the model performance results, print summary either to sdout or file
# ----------------------------------------------------------------------------------------------------------------------------

def results_analysis(file_path, output_path=None):
    df = pd.read_csv(file_path, sep='\t')  # Load data from a CSV file (tab delimited)
    print("Columns in the file:", df.columns)

    # Group data by 'dataset', 'embedding', 'model', 'wc-supervised', 'measure' and get the maximum 'value'
    result = df.groupby(['dataset', 'embeddings', 'model', 'wc-supervised', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df, result, how='inner', on=['dataset', 'embeddings', 'model', 'wc-supervised', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'embeddings', 'model', 'wc-supervised', 'measure', 'value'])

    # Specify the column order
    columns_order = ['dataset', 'model', 'pretrained', 'embeddings', 'wc-supervised', 'measure', 'params', 'tunable', 'value', 'run', 'epoch']

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in columns_order if col not in unique_result.columns]
    if missing_columns:
        print(f"Missing columns in DataFrame: {missing_columns}")
        return  # Exit the function if there are missing columns

    # Sort the DataFrame
    final_result = unique_result[columns_order].copy()
    final_result.sort_values(by=['dataset', 'model', 'pretrained', 'embeddings', 'wc-supervised', 'measure'], inplace=True)

    # Format the table
    formatted_table = tabulate(final_result, headers='keys', tablefmt='pretty', showindex=False)
    
    # Split formatted table into lines
    lines = formatted_table.split('\n')

    # Determine the length of the header line to set the separator's length
    header_line_length = len(lines[1])

    # Add separators between groups based on changes in key columns (excluding 'measure')
    grouped_lines = [lines[0], lines[1], '-' * header_line_length]  # Start with header, underline, and extra separator
    last_values = None
    
    for i, row in enumerate(final_result.itertuples(index=False)):
        # Access using index instead of attribute names to avoid potential attribute errors
        current_values = (row[0], row[1], row[2], row[3], row[4])  # dataset, model, pretrained, embeddings, wc-supervised
        if last_values and current_values != last_values:
            grouped_lines.append('-' * header_line_length)  # Use a separator as wide as the header
        last_values = current_values
        line_index = i + 3  # Offset to align with the actual content in lines, adjusted by the extra line separator
        grouped_lines.append(lines[line_index])

    final_formatted_table = '\n'.join(grouped_lines)

    # Generate output
    if output_path:
        with open(output_path, 'w') as f:
            f.write(final_formatted_table)
        print(f"Output saved to {output_path}")
    else:
        print(final_formatted_table)

    
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
