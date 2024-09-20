import pandas as pd
from tabulate import tabulate
import os
import plotly.express as px
import argparse
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt



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

    print("Generating separate charts per model and dataset...")

    print("filtering for measures:", measures)

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
                plt.figure(figsize=(14, 8))

                # Filter the dataframe for the current dataset, model, and measure
                subset_df = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['model'] == model)].copy()

                if subset_df.empty:
                    print(f"No data available for {measure}, {model}, in dataset {dataset}")
                    continue

                # Combine embeddings, representation, and dimensions into a single label for x-axis
                subset_df.loc[:, 'embedding_rep_dim'] = subset_df.apply(
                    lambda row: f"{row['embeddings']}-{row['representation']}:{row['dimensions']}", axis=1
                )

                # Sort by embeddings and then by dimensions in descending order (highest dimension first)
                subset_df = subset_df.sort_values(by=['embeddings', 'dimensions'], ascending=[True, False])

                # Dynamically adjust the palette to match the number of unique embeddings
                unique_embeddings = subset_df['embeddings'].nunique()
                color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

                # Create a bar plot
                sns.barplot(
                    data=subset_df,
                    x='embedding_rep_dim',
                    y='value',
                    hue='embeddings',
                    palette=color_palette
                )

                # Customize plot
                plt.title(f"{dataset}-{model}:{measure}", fontsize=15, weight='bold')
                plt.xlabel("Embeddings-Representation:Dimensions", fontsize=9)
                plt.ylabel(measure, fontsize=10)
                plt.ylim(y_axis_threshold, 1)  # Set y-axis range
                plt.xticks(rotation=45, ha='right', fontsize=8)  # Rotate x-axis labels at an angle
                plt.yticks(fontsize=9)  # Set y-axis font size
                plt.legend(title="Embeddings", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=10)
                plt.tight_layout()

                # Save the plot with today's date and 'matplotlib' in the filename
                plot_file_name = f"{dataset}_{measure}_{model}_{today}_matplotlib.png"
                plt.savefig(os.path.join(output_path, plot_file_name))
                print(f"Saved plot to {output_path}/{plot_file_name}")

                if (show_charts):
                    plt.show()



import os
import plotly.express as px

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
    parser.add_argument('-o', '--output_dir', type=str, default='../out', help='Directory to save the output files, default is "../out"')
    parser.add_argument('-c', '--charts', action='store_true', default=True, help='Generate charts')
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
            results_analysis(df, args.output_dir)

        if args.charts:
            
            """
            generate_charts_plotly(
                df, 
                args.output_dir, 
                show_charts=args.show, 
                y_axis_threshold=args.ystart
            )
            """
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
