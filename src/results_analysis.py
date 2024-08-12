import pandas as pd
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.express as px


def read_file(file_path):

    df = pd.read_csv(file_path, sep='\t')  # Load data from a CSV file (tab delimited)

    print("Columns in the file:", df.columns)

    return df



# ----------------------------------------------------------------------------------------------------------------------------
# results_analysis()
#
# analyze the model performance results, print summary either to sdout or file
# ----------------------------------------------------------------------------------------------------------------------------

def results_analysis(file_path, output_path=None):

    df = read_file(file_path=file_path)

    # Group data by 'dataset', 'embedding', 'model', 'wc-supervised', 'measure' and get the maximum 'value'
    result = df.groupby(['dataset', 'embeddings', 'model', 'wc_supervised', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df, result, how='inner', on=['dataset', 'embeddings', 'model', 'wc_supervised', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'embeddings', 'model', 'wc_supervised', 'measure', 'value'])

    # Specify the column order
    columns_order = ['dataset', 'model', 'pretrained', 'embeddings', 'wc_supervised', 'measure', 'params', 'tunable', 'value', 'run', 'epoch', 'cpus', 'gpus', 'mem', 'timelapse']

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in columns_order if col not in unique_result.columns]
    if missing_columns:
        print(f"Missing columns in DataFrame: {missing_columns}")
        return  # Exit the function if there are missing columns

    # Sort the DataFrame
    final_result = unique_result[columns_order].copy()
    final_result.sort_values(by=['dataset', 'model', 'pretrained', 'embeddings', 'wc_supervised', 'measure'], inplace=True)

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
        with open(output_path, 'w') as f:
            f.write(final_formatted_table)
        print(f"Output saved to {output_path}")
    else:
        print(final_formatted_table)

    
def generate_charts_seaborn(file_path, output_path=None):
    df = read_file(file_path=file_path)

    # List of measures to analyze
    measures = ['final-te-macro-F1', 'final-te-micro-F1', 'te-hamming-loss', 'te-jacard-index']

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define a vibrant and distinct color palette
    color_palette = sns.color_palette("husl", n_colors=10)  # 'husl' palette is vibrant and better spaced

    for measure in measures:
        for supervised in sorted(df['wc_supervised'].unique(), reverse=True):
            plt.figure(figsize=(14, 7))  # Adjust the figure size

            # Filter the DataFrame for the current measure and wc_supervised status
            subset_df = df[(df['measure'] == measure) & (df['wc_supervised'] == supervised)]

            # Check if the subset DataFrame is empty
            if subset_df.empty:
                print(f"No data available for {measure} with wc_supervised={supervised}")
                continue

            # Create the plot using Seaborn's barplot
            sns.barplot(data=subset_df, x='model', y='value', hue='embeddings', palette=color_palette,
                        ci=None, edgecolor='gray', linewidth=0.5)

            plt.title(f'Comparison of {measure} by Model and Embeddings - {"Supervised" if supervised else "Not Supervised"}')
            plt.ylabel('Value')
            plt.xlabel('Model')
            plt.legend(title="Embeddings", loc='upper right', bbox_to_anchor=(1.15, 1), borderaxespad=0.)

            # Save each plot in the specified output directory
            if output_path:
                plot_file_name = f"{measure}_{'supervised' if supervised else 'unsupervised'}_comparison_seaborn.png"
                plot_file = os.path.join(output_path, plot_file_name)
                plt.savefig(plot_file, bbox_inches='tight')  # Ensure everything is included in the saved image
                print(f"Saved plot for {measure} at {plot_file}")

            # Show plot
            plt.show()

            plt.close()


def generate_charts(file_path, output_path=None):

    df = read_file(file_path=file_path)

    # List of measures to analyze
    measures = ['final-te-macro-F1', 'final-te-micro-F1', 'te-hamming-loss', 'te-jacard-index']

    # Define colors for differentiation
    colors = ['lightblue', 'lightgreen', 'salmon', 'wheat', 'lightgrey', 'cyan', 'magenta', 'yellow', 'orange', 'lime']

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    for measure in measures:
        fig, ax = plt.subplots(figsize=(14, 7))  # Adjust the figure size

        legend_labels = []

        for idx, supervised in enumerate([False, True]):  # Ensure False is first for consistent order
            # Filter the DataFrame for the current measure and wc_supervised status
            measure_df = df[(df['measure'] == measure) & (df['wc_supervised'] == supervised)]

            # Check if the measure data is empty
            if measure_df.empty:
                print(f"No data available for {measure} with wc_supervised={supervised}")
                continue

            # Group data by 'model' and 'embeddings' and calculate the average of the 'value'
            result = measure_df.groupby(['model', 'embeddings'])['value'].mean().unstack()

            # Plotting side by side within the same subplot
            bars = result.plot(kind='bar', ax=ax, legend=False, color=colors[idx*len(colors)//2:(idx+1)*len(colors)//2], position=idx, width=0.35)

            # Append legend labels for each embedding type
            for embedding in result.columns:
                legend_labels.append(f'{embedding} ({"Supervised" if supervised else "Not Supervised"})')

        ax.set_title(f'Comparison of {measure} by Model and Embeddings')
        ax.set_ylabel('Value')
        ax.set_xlabel('Model')

        # Set custom legend on the left side outside the plot
        ax.legend(legend_labels, title="Embeddings and Supervision", bbox_to_anchor=(-0.15, 0.5), loc='center left', fontsize='small', borderaxespad=0.)

        # Save each plot in the specified output directory
        if output_path:
            plot_file = os.path.join(output_path, f"{measure}_comparison.png")
            plt.savefig(plot_file, bbox_inches='tight')  # Ensure everything is included in the saved image
            print(f"Saved plot for {measure} at {plot_file}")

        # Show plot
        plt.show()

        plt.close()


def generate_charts_plotly(file_path, output_path=None):
    df = read_file(file_path=file_path)

    # Filter to only include specific measures
    measures = ['final-te-macro-F1', 'final-te-micro-F1']
    df = df[df['measure'].isin(measures)]

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Using a color-blind-friendly palette
    color_palette = px.colors.qualitative.Safe

    for measure in measures:
        for supervised in sorted(df['wc_supervised'].unique(), reverse=True):
            # Filter the DataFrame for the current measure and wc_supervised status
            subset_df = df[(df['measure'] == measure) & (df['wc_supervised'] == supervised)]

            # Check if the subset DataFrame is empty
            if subset_df.empty:
                print(f"No data available for {measure} with wc_supervised={supervised}")
                continue

            # Aggregate to find the maximum value per model and embeddings
            max_df = subset_df.groupby(['model', 'embeddings']).agg({'value': 'max'}).reset_index()

            # Create the plot using Plotly Express
            fig = px.bar(max_df, x='model', y='value', color='embeddings', barmode='group',
                         title=f'Maximum Value Comparison of {measure} by Model and Embeddings - {"Supervised" if supervised else "Not Supervised"}',
                         labels={"value": "Max Value", "model": "Model"},
                         color_discrete_sequence=color_palette,
                         hover_data=['model', 'embeddings'])

            fig.update_layout(legend_title_text='Embeddings')
            fig.update_xaxes(title_text='Model')
            fig.update_yaxes(title_text='Maximum Measure Value', range=[0, max_df['value'].max() * 1.1])  # Adjust the y-axis range to fit max value

            # Save each plot in the specified output directory and show it
            if output_path:
                plot_file_name = f"max_{measure}_{'supervised' if supervised else 'unsupervised'}_comparison_plotly.html"
                plot_file = os.path.join(output_path, plot_file_name)
                fig.write_html(plot_file)  # Save as HTML to retain interactivity
                print(f"Saved interactive plot for {measure} at {plot_file}")

            fig.show()  # This will display the plot in the notebook or a web browser



# ----------------------------------------------------------------------------------------------------------------------------
#     

def main():
    if len(sys.argv) < 2:
        print("Usage: python results_analysis.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    #results_analysis(input_file, output_file)

    #generate_charts(input_file, output_file)

    #generate_charts_seaborn(input_file, output_file)

    generate_charts_plotly(input_file, output_file)


if __name__ == "__main__":
    main()
