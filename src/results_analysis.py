import pandas as pd
import sys
from tabulate import tabulate
import os
import plotly.express as px
import argparse



Y_AXIS_THRESHOLD = 0.5               # when to start the Y axis to show differentiation in the plot

# ----------------------------------------------------------------------------------------------------------------------------
# results_analysis()
#
# analyze the model performance results, print summary either to sdout or file
# ----------------------------------------------------------------------------------------------------------------------------

def results_analysis(df, output_path='../out'):

    print("analyzing results...")

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Group data by 'dataset', 'embedding', 'model', 'wc-supervised', 'measure' and get the maximum 'value'
    result = df.groupby(['dataset', 'embeddings', 'model', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df, result, how='inner', on=['dataset', 'embeddings', 'model', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'embeddings', 'model', 'measure', 'value'])

    # Specify the column order
    columns_order = ['dataset', 'model', 'pretrained', 'embeddings', 'measure', 'params', 'tunable', 'value', 'run', 'epoch', 'cpus', 'gpus', 'mem', 'timelapse']

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in columns_order if col not in unique_result.columns]
    if missing_columns:
        print(f"Missing columns in DataFrame: {missing_columns}")
        return  

    # Sort the DataFrame
    final_result = unique_result[columns_order].copy()
    final_result.sort_values(by=['dataset', 'model', 'pretrained', 'embeddings', 'measure'], inplace=True)

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

    # Add a final border line
    grouped_lines.append('-' * header_line_length)

    final_formatted_table = '\n'.join(grouped_lines)

    # Generate output
    if output_path:

        file_name = "results_analysis.out"
        output_file = os.path.join(output_path, "results_analysis.out")
        
        with open(output_file, 'w') as f:
            f.write(final_formatted_table)
        
        print(f"Output saved to {output_file}")
    else:
        print(final_formatted_table)


# ----------------------------------------------------------------------------------------------------------------------------
# generate_charts_plotly()
#
# Generate plots for models and their performance evalutaion metrics and save to file 
# ----------------------------------------------------------------------------------------------------------------------------

def generate_charts_plotly(df, output_path='../out', show_charts=False, y_axis_threshold=Y_AXIS_THRESHOLD):

    print("generating charts to output directory:", output_path)

    # Filter to only include specific measures
    measures = ['final-te-macro-F1', 'final-te-micro-F1']

    print("filtering for measures:", measures)

    df_measures = df[df['measure'].isin(measures)]
    df_timelapse = df[['dataset', 'model', 'embeddings', 'timelapse']].drop_duplicates()

    print("df shape after filtering for measures:", df_measures.shape)
    print("df shape after filtering for timelapse:", df_timelapse.shape)

    if df_measures.empty and df_timelapse.empty:
        print("Error: No data available for the specified measures or timelapse")
        return

    print("df_measures shape after filtering:", df_measures.shape)
    print("df_measures:\n", df_measures.head())

    if df_measures.empty:
        print("Error: No data available for the specified measures")
        return

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Using a color-blind-friendly palette
    color_palette = px.colors.qualitative.Safe

    # Generate charts for different measures and datasets
    for measure in measures:
        for dataset in df['dataset'].unique():
            print(f"generating plots for {measure} in dataset {dataset}...")

            subset_df = df_measures[(df_measures['measure'] == measure) & (df_measures['dataset'] == dataset)]

            if subset_df.empty:
                print(f"No data available for {measure} in dataset {dataset}")
                continue

            # Print to see if any data is missing
            print(f"Subset for {dataset}:\n", subset_df)

            max_df = subset_df.groupby(['model', 'embeddings']).agg({'value': 'max'}).reset_index()
            max_df['value'] = pd.to_numeric(max_df['value'], errors='coerce').fillna(0)  # Convert to numeric

            title_text = f'Dataset: {dataset.upper()}; Measure: {measure} [by Model and Embeddings Type]'

            fig = px.bar(max_df, x='model', y='value', color='embeddings', barmode='group',
                         title=title_text,
                         labels={"value": "Max Value" if measure != 'timelapse' else "Average Time (s)", "model": "Model"},
                         color_discrete_sequence=color_palette,
                         hover_data=['model', 'embeddings'])

            fig.update_layout(
                title={
                    'text': title_text,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(
                        family="Arial",
                        size=16,
                        color='black',
                        weight='bold'
                    )
                },
                legend_title_text='Embeddings'
            )

            fig.update_xaxes(title_text='Model')
            if measure != 'timelapse':
                fig.update_yaxes(title_text='Maximum Measure Value', range=[min(y_axis_threshold, max_df['value'].min()), max_df['value'].max() * 1.1])
            else:
                fig.update_yaxes(title_text='Average Time (seconds)', range=[0, max_df['value'].max() * 1.1])

            # Save each plot in the specified output directory
            if output_path:
                print(f"Generating interactive plot for {measure} in dataset {dataset}...")

                plot_file_name = f"{dataset}_{measure}_pretrained.html"
                plot_file = os.path.join(output_path, plot_file_name)
                fig.write_html(plot_file)

                print(f"Saved interactive plot for {measure} in dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()

    print("df_timelapse shape after filtering:", df_timelapse.shape)
    print("df_timelapse:", df_timelapse.head())

    if df_timelapse.empty:
        print("Error: No data available for timelapse analysis.")
        return  

    # Generate charts for timelapse (time taken by each model)
    for dataset in df_timelapse['dataset'].unique():
        print(f"generating timelapse plots for dataset {dataset}...")

        subset_df = df_timelapse[df_timelapse['dataset'] == dataset]

        if subset_df.empty:
            print(f"No timelapse data available for dataset {dataset}")
            continue

        # Aggregate to find the average timelapse per model and embeddings
        avg_timelapse_df = subset_df.groupby(['model', 'embeddings']).agg({'timelapse': 'mean'}).reset_index()

        title_text = f'Dataset: {dataset.upper()}; Timelapse [by Model and Embeddings Type]'

        # Create the plot using Plotly Express
        fig = px.bar(avg_timelapse_df, x='model', y='timelapse', color='embeddings', barmode='group',
                     title=title_text,
                     labels={"timelapse": "Average Time (seconds)", "model": "Model"},
                     color_discrete_sequence=color_palette,
                     hover_data=['model', 'embeddings'])

        fig.update_layout(
            title={
                'text': title_text,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    family="Arial",
                    size=16,
                    color='black',
                    weight='bold'
                )
            },
            legend_title_text='Embeddings'
        )
        
        fig.update_xaxes(title_text='Model')
        fig.update_yaxes(title_text='Average Time (seconds)', range=[0, avg_timelapse_df['timelapse'].max() * 1.1])

        # Save each plot in the specified output directory
        if output_path:
            print(f"Generating interactive plot for timelapse in dataset {dataset}...")

            plot_file_name = f"{dataset}_timelapse_pretrained.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved interactive plot for timelapse in dataset {dataset} at {plot_file}")

        if show_charts:
            fig.show()


            

# ----------------------------------------------------------------------------------------------------------------------------
# read_data_file()
#
# read data file and put it in a DataFrame object. If doesn't exist, return null 
# ----------------------------------------------------------------------------------------------------------------------------

def read_data_file(file_path=None, debug=False):

    if (debug):
        print("Reading data file:", file_path)

    if not (os.path.exists(file_path)):
        print("Error: File not found:", file_path) 
        return None
       
    df = pd.read_csv(file_path, sep='\t')  # Load data from a CSV file (tab delimited)

    if (debug):
        print("Columns in the file:", df.columns)

    return df



# ----------------------------------------------------------------------------------------------------------------------------
# main()
#
# Read arguments and call results_analysis or generate_charts_plotly accordingly 
# ----------------------------------------------------------------------------------------------------------------------------

def main():

    print("----- Results Analysis -----")
    
    parser = argparse.ArgumentParser(description="Analyze model results and generate charts or summaries")

    parser.add_argument('file_path', type=str, help='Path to the CSV file with the data')
    parser.add_argument('-output_dir', type=str, default='../out', help='Directory to save the output files, default is "../out"')
    
    parser.add_argument('-c', '--charts', action='store_true', help='Generate charts')
    parser.add_argument('-s', '--summary', action='store_true', help='Generate summary')
    parser.add_argument('-d', action='store_true', default=False, help='debug mode')
    
    parser.add_argument('-ystart', type=float, default=Y_AXIS_THRESHOLD, help='Y-axis starting value for the charts (default: 0.6)')

    parser.add_argument('-show', action='store_true', help='Display charts interactively (requires -c)')

    args = parser.parse_args()

    # Ensure at least one operation is specified
    if not (args.charts or args.summary):
        parser.error("No action requested, add -c for charts or -s for summary")

    debug = args.d
    print("debug mode:", debug)

    print("y_start:", args.ystart)

    df = read_data_file(args.file_path, debug=debug)

    if df is not None:

        if (debug):
            print("Data file read successfully, df:", df.shape)

        if args.summary:
            results_analysis(df, args.output_dir)

        if args.charts:
            generate_charts_plotly(df, args.output_dir, show_charts=args.show, y_axis_threshold=args.ystart)
    else:
        print("Error: Data file not found or empty")



if __name__ == "__main__":
    main()
