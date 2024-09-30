import pandas as pd
from tabulate import tabulate
import os
import plotly.express as px
import argparse
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np


import os
import plotly.express as px

import imgkit


# measures filter: report on these specific measures
measures = ['final-te-macro-F1', 'final-te-micro-F1']



Y_AXIS_THRESHOLD = 0.3               # when to start the Y axis to show differentiation in the plot

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

    # Filter for Macro and Micro F1 scores only
    df_filtered = df[df['measure'].isin(['final-te-macro-F1', 'final-te-micro-F1'])]

    # Group data by 'dataset', 'model', 'embeddings', 'representation', 'measure'
    result = df_filtered.groupby(['dataset', 'model', 'embeddings', 'representation', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df_filtered, result, how='inner', on=['dataset', 'model', 'embeddings', 'representation', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'model', 'embeddings', 'representation', 'measure', 'value'])

    # Specify the column order
    columns_order = ['class_type', 'comp_method', 'model', 'dataset', 'embeddings', 'representation', 'dimensions', 'measure', 'value', 'optimized', 'timelapse', 'run', 'epoch', 'os', 'cpus', 'gpus', 'mem']		

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in columns_order if col not in unique_result.columns]
    if missing_columns:
        print(f"Missing columns in DataFrame: {missing_columns}")
        return  

    # Sort the DataFrame
    final_result = unique_result[columns_order].copy()
    final_result.sort_values(by=['dataset', 'model', 'embeddings', 'representation', 'measure'], inplace=True)

    print("final result:\n", final_result)

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
        current_values = (row[0], row[1], row[2], row[3], row[4])  # dataset, model, embeddings, representation, measure
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
        output_file = os.path.join(output_path, file_name)
        
        with open(output_file, 'w') as f:
            f.write(final_formatted_table)
        
        print(f"Output saved to {output_file}")
    else:
        print(final_formatted_table)






def generate_charts_matplotlib(df, output_path='../out', y_axis_threshold=Y_AXIS_THRESHOLD, show_charts=False, debug=False):
    """
    The generate_charts_matplotlib function generates bar charts for each combination of model, dataset, and measure from a 
    given DataFrame. It uses Matplotlib and Seaborn to create plots that are colorblind-friendly, showing the performance of 
    models based on a specific measure for a given dataset.

    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - y_axis_threshold (default: Y_AXIS_THRESHOLD): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: False): Boolean flag to control whether the charts are displayed interactively.
    - debug (default: False): Boolean flag to print additional debug information during execution.

    Returns:
    - None: The function saves the generated plots as PNG files in the specified output
    """

    print("Generating separate charts per model and dataset...")

    print("Filtering for measures:", measures)
    
    # Filter for the specific measures of interest
    df_measures = df[df['measure'].isin(measures)]
    print("df shape after filtering for measures:", df_measures.shape)
    if df_measures.empty:
        print("Error: No data available for the specified measures")
        return

    # Set up a colorblind-friendly color palette and plot style
    sns.set(style="whitegrid")

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')

    for measure in measures:
        for dataset in df['dataset'].unique():
            for model in df['model'].unique():
                # Increase the figure size for better visibility
                plt.figure(figsize=(20, 12))  # Larger figure size

                # Filter the dataframe for the current dataset, model, and measure
                subset_df = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['model'] == model)].copy()

                if subset_df.empty:
                    print(f"No data available for {measure}, {model}, in dataset {dataset}")
                    continue

                # Combine embeddings, representation, and dimensions into a single label for the x-axis
                subset_df['embedding_rep_dim'] = subset_df.apply(
                    lambda row: f"{row['embeddings']}-{row['representation']}:{row['dimensions']}", axis=1
                )

                # Sort by dimensions in descending order (highest dimension first)
                subset_df = subset_df.sort_values(by='dimensions', ascending=False)

                # Dynamically adjust the palette to match the number of unique embeddings
                unique_embeddings = subset_df['embeddings'].nunique()
                color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

                # Create a bar plot
                sns.barplot(
                    data=subset_df,
                    x='embedding_rep_dim',  # Use the combined field with embeddings, representation, and dimensions
                    y='value',
                    hue='embeddings',
                    palette=color_palette,
                    order=subset_df['embedding_rep_dim']  # Explicitly set the order based on sorted dimensions
                )

                # Customize plot
                plt.title(f"{dataset}-{model}:{measure}", fontsize=20, weight='bold')  # Increased title size
                plt.xlabel("Embeddings-Representation:Dimensions", fontsize=14)  # Larger x-axis label
                plt.ylabel(measure, fontsize=14)  # Larger y-axis label
                plt.ylim(y_axis_threshold, 1)  # Set y-axis range

                # Adjust y-axis ticks for more granularity (twice as many ticks, e.g., every 0.05)
                plt.yticks(np.arange(y_axis_threshold, 1.01, 0.05), fontsize=10, fontweight='bold')  # Smaller, bold y-axis labels

                # Change x-axis label font style (smaller, bold)
                plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')  # Smaller, bold font for x-axis labels
                
                # Customize the legend
                plt.legend(title="Embeddings", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)  # Larger legend
                plt.tight_layout()

                # Save the plot with today's date and 'matplotlib' in the filename
                plot_file_name = f"{dataset}_{measure}_{model}_{today}_matplotlib.png"
                plt.savefig(os.path.join(output_path, plot_file_name), dpi=300)  # Increased DPI for better resolution
                print(f"Saved plot to {output_path}/{plot_file_name}")

                # Optionally display the plot
                if show_charts:
                    plt.show()






def generate_charts_plotly(df, output_path='../out', y_axis_threshold=0, show_charts=True, debug=False):

    # Define the measures of interest
    measures_of_interest = ['final-te-macro-F1', 'final-te-micro-F1']

    # Filter the dataframe for the measures of interest right away
    df = df[df['measure'].isin(measures_of_interest)]
    
    if df.empty:
        print("No data available for the specified measures")
        return

    print("generating plotly charts to output directory:", output_path)

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Generate a separate chart for each measure within each dataset
    for dataset in df['dataset'].unique():

        print("processing dataset:", dataset)

        for measure in measures_of_interest:

            print(f"Generating plots for dataset {dataset} with measure {measure}...")

            # Filter the dataframe for the current dataset and measure
            subset_df1 = df[(df['dataset'] == dataset) & (df['measure'] == measure)]
            print("subset_df1:\n", subset_df1)

            # Group by representation, model, and dimensions to find maximum values
            subset_df2 = subset_df1.groupby(['representation', 'model', 'dimensions', 'embeddings']).agg({'value': 'max'}).reset_index()

            if subset_df2.empty:
                print(f"No data available for dataset {dataset} with measure {measure}")
                continue

            # Create a new column that appends the dimensions to the representation label
            #subset_df['representation_with_dim'] = subset_df['representation'] + '{' + subset_df['dimensions'].astype(str) + '}'

            # Sorting by representation
            subset_df2.sort_values(by=['representation'], inplace=True)
            print("subset_df2:\n", subset_df2)

            # Define the sorted order for the x-axis
            model_rep_order = subset_df2['representation'].tolist()

            # Create the plot, coloring by model and spreading bars on the x-axis
            fig = px.bar(subset_df2, 
                         x='representation',  # Use representation for x-axis
                         y='value', 
                         color='model',  # Color by model
                         title=f'{measure} Performance Comparison on {dataset}',
                         labels={"value": "Performance Metric", "representation": "Representation"},
                         hover_data=['representation', 'value'],                # Include representation and value in hover
                         category_orders={"representation": model_rep_order})  # Explicit sorting order

            # Adjust layout to ensure proper alignment and equal spacing, and add legend at the top-right
            fig.update_layout(
                title={
                    'text': f'{measure} Performance Across Models and Representations on {dataset}',
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
                legend_title_text='Model',
                legend=dict(
                    orientation="v",
                    y=1,
                    x=1,
                    xanchor='right',
                    yanchor='top'
                ),
                bargap=0.2,  # Increase spacing between bars
            )

            # Ensure the x-axis is treated as categorical and sorted
            fig.update_xaxes(
                title_text='Model - Representation (Dimensions)',
                type='category'  # Treat x-axis as categorical to prevent reordering
            )

            fig.update_yaxes(
                title_text='Performance Metric', 
                range=[y_axis_threshold, 1]
            )

            # Save the plot in the specified output directory
            plot_file_name = f"{dataset}_{measure}_performance_comparison.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved plot for {measure} on dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()

    print("plotly charts generation completed.")

    return df




def gen_timelapse_plots(df, output_path='../out', show_charts=False, debug=False):

    print("generating timelapse plots...")

    # Ensure that only relevant columns are used
    df_timelapse = df[['dataset', 'model', 'embeddings', 'representation', 'dimensions', 'timelapse']].drop_duplicates()
    print("df shape after filtering for timelapse:", df_timelapse.shape)

    if df_timelapse.empty:
        print("Error: No data available for timelapse analysis.")
        return

    # Generate charts for timelapse (time taken by each model and representation)
    for dataset in df_timelapse['dataset'].unique():
        print(f"Generating timelapse plots for dataset {dataset}...")

        # Explicitly copy the subset to avoid SettingWithCopyWarning
        subset_df = df_timelapse[df_timelapse['dataset'] == dataset].copy()

        if subset_df.empty:
            print(f"No timelapse data available for dataset {dataset}")
            continue

        # Sort by dimensions in descending order (highest dimension first)
        subset_df = subset_df.sort_values(by='dimensions', ascending=False)
        #print("subset_df:\n", subset_df)

        # Create a new column to append the dimensions to the representation label
        subset_df['representation_with_dim'] = subset_df['representation'] + ' (' + subset_df['dimensions'].astype(str) + ')'

        # Aggregate to find the average timelapse per model, embeddings, and representation_with_dim
        avg_timelapse_df = subset_df.groupby(['model', 'embeddings', 'representation_with_dim']).agg({'timelapse': 'mean'}).reset_index()

        # Dynamically adjust the palette to match the number of unique embeddings
        unique_vals = avg_timelapse_df['embeddings'].nunique()
        print(f"unique_vals (embeddings): {unique_vals}")

        # Use Seaborn palette and convert it to a list of hex colors for Plotly
        color_palette = sns.color_palette("colorblind", n_colors=unique_vals).as_hex()

        # Get the sorted order of representations with dimensions
        sorted_representation_with_dim = subset_df['representation_with_dim'].tolist()

        title_text = f'Dataset: {dataset.upper()}; Timelapse [by Model, Embeddings, and Representation]'

        # Create the plot using Plotly Express, coloring by embeddings
        fig = px.bar(avg_timelapse_df, 
                     x='representation_with_dim',  # Representations with dimensions on the x-axis
                     y='timelapse', 
                     color='embeddings',  # Color by embeddings
                     barmode='group',
                     title=title_text,
                     labels={'timelapse': "Average Time (seconds)", 'representation_with_dim': "Representation (Dimensions)"},
                     color_discrete_sequence=color_palette,                                                                         # Use the corrected color palette
                     hover_data=['model', 'embeddings', 'representation_with_dim'],                                                 # Include model, embeddings, and representation in hover
                     category_orders={"representation_with_dim": sorted_representation_with_dim}                                    # Enforce sorted x-axis order
                )

        # Set up the layout, including the legend and x-axis configuration
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
            legend_title_text='Embeddings',  # Set legend title for embeddings
            legend=dict(
                orientation="v",
                y=1,
                x=1,
                xanchor="right",
                yanchor="top"
            ),
            bargap=0.2  # Control space between bars
        )

        # Angle the x-axis labels to make them easier to read
        fig.update_xaxes(title_text='Representation (Dimensions)', tickangle=-45)  # Rotate the labels for readability
        fig.update_yaxes(title_text='Average Time (seconds)', range=[0, avg_timelapse_df['timelapse'].max() * 1.1])

        # Save each plot in the specified output directory
        if output_path:
            plot_file_name = f"{dataset}_timelapse_by_representation_with_dimensions.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved interactive plot for timelapse in dataset {dataset} at {plot_file}")

        if show_charts:
            fig.show()

    print("Timelapse plots generation completed.")

    return df_timelapse





def generate_grouped_tables(df, output_dir):
    # Split the 'mode' column into separate columns for dataset, model, mode, and mix
    df[['Dataset', 'Model', 'Mix', 'comp_method']] = df['mode'].str.split(':', expand=True)

    # Filter the dataframe for the required measures
    measures = ['final-te-macro-F1', 'final-te-micro-F1']
    filtered_df = df[df['measure'].isin(measures)]
    print(filtered_df)

    # Iterate through each dataset and model to generate tables
    for (dataset, model), group_df in filtered_df.groupby(['Dataset', 'Model']):
        output_html = f"{output_dir}/{dataset}_{model}_results.html"
        output_csv = f"{output_dir}/{dataset}_{model}_results.csv"
        render_grouped_table_with_pandas(group_df, dataset, model, output_html, output_csv)


def render_grouped_table_with_pandas(dataframe, dataset, model, output_html, output_csv):
    # Group the data by embeddings and mix (formerly Mode) within each embedding
    grouped = dataframe.groupby(['embeddings', 'Mix', 'class_type'], as_index=False)

    # Select only the required columns, including the new 'dimensions' column
    selected_columns = ['embeddings', 'Mix', 'comp_method', 'representation', 'dimensions', 'measure', 'value', 'timelapse']

    # Create an HTML table manually, ensuring that embeddings and mix only display once per group
    rows = []
    previous_embeddings = None
    previous_mix = None

    # Prepare CSV data
    csv_rows = [['embeddings', 'mix', 'comp_method', 'representation', 'dimensions', 'measure', 'value', 'timelapse (seconds)']]

    for (embeddings, mix, class_type), group in grouped:
        # Determine if we need a bold line for the first embeddings group
        group_border = "border-top: 3px solid black;" if embeddings != previous_embeddings else ""

        first_row = True
        for _, row in group.iterrows():
            # Format value to 3 decimal places and timelapse with comma separator
            formatted_value = f"{row['value']:.3f}"
            formatted_timelapse = f"{row['timelapse']:,.0f}"

            # Prepare the HTML row
            if first_row:
                # Display embeddings in bold and mix in italics, apply the bold border for new embeddings group
                row_html = f"<tr style='font-size: 12px; {group_border}'><td><b>{row['embeddings']}</b></td><td><i>{row['Mix']}</i></td>"
                first_row = False
            else:
                # Leave the embeddings and mix columns empty for subsequent rows, apply dotted line between Mix combinations
                dotted_border = "border-bottom: 1px dotted gray;" if mix != previous_mix else ""
                row_html = f"<tr style='font-size: 12px; {dotted_border}'><td></td><td></td>"

            # Add the rest of the columns (comp_method, representation, dimensions, measure, formatted value, formatted timelapse)
            row_html += f"<td>{row['comp_method']}</td><td>{row['representation']}</td><td>{row['dimensions']}</td><td>{row['measure']}</td><td>{formatted_value}</td><td>{formatted_timelapse}</td></tr>"
            rows.append(row_html)

            # Prepare the CSV row
            csv_row = [row['embeddings'], row['Mix'], row['comp_method'], row['representation'], row['dimensions'], row['measure'], formatted_value, formatted_timelapse]
            csv_rows.append(csv_row)

        # Update previous_embeddings and previous_mix to track the current group
        previous_embeddings = embeddings
        previous_mix = mix

    # Join all rows together and style the table with smaller columns and fit width
    table_html = """
    <table border='1' style='border-collapse: collapse; font-size: 12px; table-layout: fixed; width: 100%;'>
    <colgroup>
        <col style='width: 8%;'>
        <col style='width: 8%;'>
        <col style='width: 12%;'>
        <col style='width: 12%;'>
        <col style='width: 10%;'> <!-- Dimensions column -->
        <col style='width: 15%;'>
        <col style='width: 10%;'>
        <col style='width: 10%;'>
    </colgroup>
    <tr><th>embeddings</th><th>mix</th><th>comp_method</th><th>representation</th><th>dimensions</th><th>measure</th><th>value</th><th>timelapse (seconds)</th></tr>
    """
    table_html += "".join(rows)
    table_html += "</table>"

    # Write the HTML table to file, including class_type in the title
    with open(output_html, 'w') as f:
        f.write(f"<h2>Results for Dataset: {dataset}, Model: {model}, Class Type: {class_type}</h2>")
        f.write(table_html)

    print(f"HTML Table saved as {output_html}.")

    # Write CSV data to file
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)

    print(f"CSV saved as {output_csv}.")










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
    parser.add_argument('-o', '--output_dir', type=str, default='../out', help='output directory for files, defaults to ../out. Used with -s (--summary) option')
    parser.add_argument('-c', '--charts', action='store_true', default=False, help='Generate charts')
    parser.add_argument('-r', '--runtimes', action='store_true', default=False, help='Generate timrlapse charts')
    parser.add_argument('-s', '--summary', action='store_true', default=False, help='Generate summary')
    parser.add_argument('-d', action='store_true', default=False, help='debug mode')
    parser.add_argument('-y', '--ystart', type=float, default=Y_AXIS_THRESHOLD, help='Y-axis starting value for the charts (default: 0.6)')
    parser.add_argument('-show', action='store_true', help='Display charts interactively (requires -c)')

    args = parser.parse_args()

    print("args: ", args)

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

            if (args.output_dir is None):
                print("Error: Output file name required with -s (--summary) option")
                return
            
            #results_analysis(df, args.output_dir)

            generate_grouped_tables(df, args.output_dir)


        if args.charts:
            
            generate_charts_plotly(
                df, 
                args.output_dir, 
                show_charts=args.show, 
                y_axis_threshold=args.ystart
            )
            
            #
            # matplotlib option is less interactive but handles more test cases - its split by dataset 
            # and model as opposed to just dataset as the plotly graphs are designed for 
            #
            generate_charts_matplotlib(
                df, 
                args.output_dir,
                show_charts=args.show,
                y_axis_threshold=args.ystart,
                debug=debug
                
            )
            
        if (args.runtimes):
            gen_timelapse_plots(
                df, 
                args.output_dir, 
                show_charts=args.show,
                debug=debug
            )
        
    else:
        print("Error: Data file not found or empty")



if __name__ == "__main__":
    main()
