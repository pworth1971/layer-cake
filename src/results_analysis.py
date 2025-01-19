import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.express as px
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
import csv

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show
from bokeh.transform import factor_cmap
from bokeh.palettes import Category20  # A palette with up to 20 colors

from util.common import OUT_DIR, WORD_BASED_MODELS, TOKEN_BASED_MODELS



# -----------------------------------------------------------------------------------------------------------------------------------
#
# measures filter: report on these specific measures
#
MEASURES = ['final-te-macro-f1', 'final-te-micro-f1']

#
# Define the measures to be included in CSV file output
#
CSV_MEASURES = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy', 'te-recall', 'te-precision']  


#
# number of results to display in performance plot (model by representation)
#
NUM_RESULTS = 80


Y_AXIS_THRESHOLD = 0.25                                     # when to start the Y axis to show differentiation in the plot
#
# -----------------------------------------------------------------------------------------------------------------------------------




def generate_charts_matplotlib(df, output_path='../out', neural=False, y_axis_threshold=Y_AXIS_THRESHOLD, show_charts=False, debug=False):
    """
    Generates combined bar charts for word-based, subword-based, and token-based models on the same chart.
    """
    print("Generating combined charts for all embeddings...")

    # Filter for measures of interest
    df_measures = df[df['measure'].isin(MEASURES)]
    if debug:
        print("df shape after filtering for measures:", df_measures.shape)

    if df_measures.empty:
        print("Error: No data available for the specified measures.")
        return

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set colorblind-friendly style
    sns.set(style="whitegrid")

    # Get today's date for file naming
    today = datetime.today().strftime('%Y-%m-%d')

    for measure in MEASURES:
        for dataset in df['dataset'].unique():
            for model in df['model'].unique():
                # Filter data for the current combination
                df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['model'] == model)].copy()

                if df_subset.empty:
                    print(f"No data available for {measure}, {model}, in dataset {dataset}.")
                    continue

                # Extract the base embeddings (everything before the colon)
                if (neural):
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

                # Combine representation and dimensions into a single label for x-axis
                df_subset['rep_dim'] = df_subset.apply(
                    lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                )

                # Sort by dimensions in descending order
                df_subset = df_subset.sort_values(by='dimensions', ascending=False)

                # Create a color palette based on unique embedding types
                unique_embeddings = df_subset['embedding_type'].nunique()
                color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

                # Create the plot
                plt.figure(figsize=(20, 12))
                sns.barplot(
                    data=df_subset,
                    x='rep_dim',
                    y='value',
                    hue='embedding_type',
                    palette=color_palette,
                    order=df_subset['rep_dim']
                )

                # Customize the plot
                plt.title(
                    f"Dataset: {dataset}, Model: {model}, Measure: {measure}",
                    fontsize=20, weight='bold'
                )
                plt.xlabel("Embeddings-Representation:Dimensions", fontsize=14)
                plt.ylabel(measure, fontsize=14)
                plt.ylim(y_axis_threshold, 1)
                plt.xticks(rotation=45, ha='right', fontsize=9, fontweight='bold')
                plt.yticks(fontsize=9, fontweight='bold')
                plt.legend(title="Embedding Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
                plt.tight_layout()

                # Save the plot
                plot_file_name = f"{dataset}_{measure}_{model}_combined_{today}.png"
                plt.savefig(os.path.join(output_path, plot_file_name), dpi=450)
                print(f"Saved plot to {output_path}/{plot_file_name}")

                # Optionally show the plot
                if show_charts:
                    plt.show()




def generate_charts_matplotlib_split(df, output_path='../out', neural=False, y_axis_threshold=Y_AXIS_THRESHOLD, show_charts=False, debug=False):
    """
    The generate_charts_matplotlib function generates bar charts for each combination of model, dataset, measure, and embedding type 
    from a given DataFrame. It uses Matplotlib and Seaborn to create plots that are colorblind-friendly, showing the performance of 
    models based on a specific measure for a given dataset and embedding type.

    It generates separate plots for word-based and token-based embeddings.

    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - y_axis_threshold (default: Y_AXIS_THRESHOLD): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: False): Boolean flag to control whether the charts are displayed interactively.
    - debug (default: False): Boolean flag to print additional debug information during execution.

    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """

    print("Generating separate charts per model, dataset, and embedding type...")

    print("Filtering for measures:", MEASURES)
    
    # Filter for the specific measures of interest
    df_measures = df[df['measure'].isin(MEASURES)]
    if (debug):
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

    for measure in MEASURES:
        for dataset in df['dataset'].unique():
            for model in df['model'].unique():
                # Filter the dataframe for the current dataset, model, and measure
                df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['model'] == model)].copy()

                if df_subset.empty:
                    print(f"No data available for {measure}, {model}, in dataset {dataset}")
                    continue

                # Extract the base embeddings (everything before the colon)
                if (neural):
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

                # Split into word-based and token-based embeddings
                word_based_df = df_subset[df_subset['embedding_type'].isin(WORD_BASED_MODELS)]
                token_based_df = df_subset[df_subset['embedding_type'].isin(TOKEN_BASED_MODELS)]

                # Function to plot the data
                def plot_data(subset_df, embedding_category):
                    if subset_df.empty:
                        print(f"No data available for {embedding_category} embeddings in {measure}, {model}, dataset {dataset}")
                        return

                    # Combine embedding type, representation, and dimensions into a single label for the x-axis
                    subset_df['rep_dim'] = subset_df.apply(
                        lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                    )

                    # Sort by dimensions in descending order (highest dimension first)
                    subset_df = subset_df.sort_values(by='dimensions', ascending=False)

                    # Dynamically adjust the palette to match the number of unique embedding types
                    unique_embeddings = subset_df['embedding_type'].nunique()
                    color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

                    # Create a bar plot using the embedding_type for hue (color coding based on embedding type)
                    plt.figure(figsize=(20, 12))  # Larger figure size
                    sns.barplot(
                        data=subset_df,
                        x='rep_dim',                          # Use the combined field with embeddings, representation, and dimensions
                        y='value',
                        hue='embedding_type',                 # Color based on the embedding type (first part before the colon)
                        palette=color_palette,
                        order=subset_df['rep_dim']            # Explicitly set the order based on sorted dimensions
                    )

                    # Customize plot
                    plt.title(f"DATASET:{dataset} | MODEL: {model} | MEASURE: {measure} [{embedding_category} Embeddings]", fontsize=20, weight='bold')           # Increased title size
                    plt.xlabel("Embeddings-Representation:Dimensions", fontsize=14)                                                     # Larger x-axis label
                    plt.ylabel(measure, fontsize=14)                                                                                    # Larger y-axis label
                    plt.ylim(y_axis_threshold, 1)                                                                                       # Set y-axis range

                    # Adjust y-axis ticks for more granularity (twice as many ticks, e.g., every 0.05)
                    plt.yticks(np.arange(y_axis_threshold, 1.01, 0.05), fontsize=9, fontweight='bold')              # Smaller, bold y-axis labels

                    # Change x-axis label font style to "Arial Narrow", smaller font size, and bold weight
                    plt.xticks(rotation=45, ha='right', fontsize=9, fontweight='bold')                              # Tighter and smaller x-axis labels
                    
                    # Customize the legend (based on the embedding_type field)
                    plt.legend(title="Embedding Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)  # Larger legend
                    plt.tight_layout()

                    # Save the plot with today's date and 'matplotlib' in the filename
                    plot_file_name = f"{dataset}_{measure}_{model}_{embedding_category}_{today}_matplotlib.png"
                    plt.savefig(os.path.join(output_path, plot_file_name), dpi=450)                                     # Increased DPI for better resolution
                    print(f"Saved plot to {output_path}/{plot_file_name}")

                    # Optionally display the plot
                    if show_charts:
                        plt.show()

                # Plot for word-based embeddings
                plot_data(word_based_df, 'Word and Subword Based Models')

                # Plot for token-based embeddings
                plot_data(token_based_df, 'Token (Transformer) Based Models')




# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def gen_timelapse_plots(df, output_path='../out', show_charts=False, debug=False):
    """
    Generate timelapse plots for each dataset, model, embeddings, representation, and dimensions
    using Bokeh for interactive visualizations.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing timelapse data.
        output_path (str): Directory to save the output files.
        show_charts (bool): Whether to display the charts interactively.
        debug (bool): Whether to enable debug mode for extra outputs.
    """
    print("Generating timelapse plots...")

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_timelapse = df[['dataset', 'model', 'embeddings', 'representation', 'dimensions', 'timelapse']].drop_duplicates()

    if df_timelapse.empty:
        print("Error: No data available for timelapse analysis.")
        return

    for dataset in df_timelapse['dataset'].unique():
        print(f"Generating timelapse plots for dataset {dataset}...")

        subset_df = df_timelapse[df_timelapse['dataset'] == dataset].copy()

        if subset_df.empty:
            print(f"No timelapse data available for dataset {dataset}")
            continue

        subset_df.sort_values(by='dimensions', ascending=False, inplace=True)
        subset_df['representation_with_dim'] = subset_df['representation'] + ' (' + subset_df['dimensions'].astype(str) + ')'
        subset_df['embeddings_prefix'] = subset_df['embeddings'].apply(lambda x: x.split(':')[0])  # Extract prefix

        avg_timelapse_df = subset_df.groupby(['model', 'embeddings_prefix', 'representation_with_dim']).agg({'timelapse': 'mean'}).reset_index()

        source = ColumnDataSource(data=avg_timelapse_df)

        num_colors = len(avg_timelapse_df['embeddings_prefix'].unique())
        palette = Category20[20][:num_colors]  # Use Category20 for better color distribution

        p = figure(
            title=f"Timelapse Data for {dataset}",                              # Title
            x_range=avg_timelapse_df['representation_with_dim'].unique(),
            height=1400,                                                        # Increased plot height
            width=1800,                                                         # Increased plot width
            tools="pan,wheel_zoom,box_zoom,reset"
        )
        p.vbar(
            x='representation_with_dim', top='timelapse', width=0.9, source=source,
            legend_field='embeddings_prefix',
            line_color='white',
            fill_color=factor_cmap('embeddings_prefix', palette=palette, factors=avg_timelapse_df['embeddings_prefix'].unique())
        )

        p.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Embeddings", "@embeddings_prefix"), ("Timelapse", "@timelapse")]))

        p.title.text_font_size = '16pt'
        p.xaxis.axis_label = "Representation (Dimensions)"
        p.yaxis.axis_label = "Average Time (seconds)"
        p.xaxis.major_label_orientation = 1
        p.xaxis.major_label_text_font_size = "8pt"
        p.legend.title = 'Embeddings'
        p.legend.label_text_font_size = '8pt'

        html_file = os.path.join(output_path, f"{dataset}_timelapse_plot_interactive.html")
        output_file(html_file)
        save(p)
        print(f"Saved interactive HTML plot for dataset {dataset} at {html_file}")

        if show_charts:
            show(p)  # Open in browser

    print("Timelapse plots generation completed.")




def gen_timelapse_plots_orig(df, output_path='../out', show_charts=False, debug=False):
    """
    gen_timelapse_plots generates timelapse plots for each dataset, model, embeddings, representation, and dimensions from a given DataFrame.
    
    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - show_charts (default: False): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during 
    
    Returns:
    - None: The function saves the generated plots as HTML files in the specified output
    """

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

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def model_performance_comparison(df, output_path='../out', neural=False, y_axis_threshold=0, show_charts=True, debug=False, num_results=None):
    """
    model_performance_comparison generates bar charts for each combination of model, dataset, and measure from a given DataFrame.
    
    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning models or classical ML models.
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during
    - num_results (default: None): Number of top results to display in the plot (None for all results).
    
    Returns:
    - None: The function saves the generated plots as HTML files in the specified output directory.
    """
    
    # Filter the dataframe for the measures of interest right away
    df = df[df['measure'].isin(MEASURES)]
    
    if df.empty:
        print("No data available for the specified measures")
        return

    print("\n\tgenerating plotly charts to output directory:", output_path)

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    if neural:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings']
    else:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings'].apply(lambda x: x.split(':')[0])

    # Generate a separate chart for each measure within each dataset
    for dataset in df['dataset'].unique():

        print("processing dataset:", dataset)

        for measure in MEASURES:

            print(f"\n\tGenerating plots for dataset {dataset} with measure {measure}...")

            # Filter the dataframe for the current dataset and measure
            subset_df1 = df[(df['dataset'] == dataset) & (df['measure'] == measure)]
            if debug:
                print("subset_df1:\n", subset_df1)

            # Group by representation, model, dimensions, and embeddings to find maximum values
            subset_df2 = subset_df1.groupby(['representation', 'model', 'dimensions', 'embeddings', 'embedding_type']).agg({'value': 'max'}).reset_index()

            if subset_df2.empty:
                print(f"No data available for dataset {dataset} with measure {measure}")
                continue

            # Update the representation column to include the dimensions in curly brackets, italicized
            subset_df2['representation'] = subset_df2.apply(lambda row: f"{row['representation']} <i>{{{row['dimensions']}}}</i>", axis=1)

            # Sort by performance value in descending order
            subset_df2.sort_values(by='value', ascending=False, inplace=True)

            # Filter to the top num_results if specified
            if num_results is not None:
                subset_df2 = subset_df2.head(num_results)

            if debug:
                print("subset_df2 sorted by value:\n", subset_df2)

            # Define the sorted order for the x-axis based on performance
            model_rep_order = subset_df2['representation'].tolist()

            # Create the plot, coloring by the embedding type (extracted earlier)
            fig = px.bar(subset_df2, 
                         x='representation', 
                         y='value', 
                         color='embedding_type',                                                                # Color by embedding type
                         title=f'{measure} Performance Comparison on {dataset}',
                         labels={"value": "Performance Metric", "representation": "Representation"},
                         hover_data=['representation', 'value', 'embedding_type'],                              # Include representation, value, and embedding type in hover
                         category_orders={"representation": model_rep_order})                                   # Explicit sorting order

            # Adjust layout to ensure proper alignment and equal spacing, and add legend at the top-right
            fig.update_layout(
                title={
                    'text': f'{dataset} {measure} model performance [by representation]',
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
                legend_title_text='Embedding Type',                                                  # Updated legend title to reflect embedding type
                legend=dict(
                    orientation="v",
                    y=1,
                    x=1,
                    xanchor='right',
                    yanchor='top'
                ),
                bargap=0.2,  # Increase spacing between bars
            )

            # Ensure the x-axis is treated as categorical and sorted, and rotate the labels
            fig.update_xaxes(
                title_text='Representation',
                type='category',                                                # Treat x-axis as categorical to prevent reordering
                tickangle=-45,                                                  # Rotate the x-axis labels
                tickfont=dict(size=9)                                           # Make the x-axis labels slightly smaller
            )

            fig.update_yaxes(
                title_text='Performance', 
                range=[y_axis_threshold, 1]
            )

            # Get the current date in YYYY-MM-DD format
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Save the plot in the specified output directory
            plot_file_name = f"{dataset}_{measure}_performance_comparison.{current_date}.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved plot for {measure} on dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()

    return df




def model_performance_comparison_all(df, output_path='../out', neural=False, y_axis_threshold=0, show_charts=True, debug=False):
    """
    model_performance_comparison generates bar charts for each combination of model, dataset, and measure from a given DataFrame.
    
    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning models or classical ML models.
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during
    - num_results (default: None): Number of top results to display in the plot (None for all results).
    
    Returns:
    - None: The function saves the generated plots as HTML files in the specified output directory.
    """
    
    # Filter the dataframe for the measures of interest right away
    df = df[df['measure'].isin(MEASURES)]
    
    if df.empty:
        print("No data available for the specified measures")
        return

    print("generating plotly charts to output directory:", output_path)

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    if neural:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings']
    else:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings'].apply(lambda x: x.split(':')[0])

    # Generate a separate chart for each measure within each dataset
    for dataset in df['dataset'].unique():

        print("processing dataset:", dataset)

        for measure in MEASURES:

            print(f"Generating plots for dataset {dataset} with measure {measure}...")

            # Filter the dataframe for the current dataset and measure
            subset_df1 = df[(df['dataset'] == dataset) & (df['measure'] == measure)]
            if debug:
                print("subset_df1:\n", subset_df1)

            # Group by representation, model, dimensions, and embeddings to find maximum values
            subset_df2 = subset_df1.groupby(['representation', 'model', 'dimensions', 'embeddings', 'embedding_type']).agg({'value': 'max'}).reset_index()

            if subset_df2.empty:
                print(f"No data available for dataset {dataset} with measure {measure}")
                continue

            # Update the representation column to include the dimensions in curly brackets, italicized
            subset_df2['representation'] = subset_df2.apply(lambda row: f"{row['representation']} <i>{{{row['dimensions']}}}</i>", axis=1)

            # Sort by performance value in descending order
            subset_df2.sort_values(by='value', ascending=False, inplace=True)

            if debug:
                print("subset_df2 sorted by value:\n", subset_df2)

            # Define the sorted order for the x-axis based on performance
            model_rep_order = subset_df2['representation'].tolist()

            # Create the plot, coloring by the embedding type (extracted earlier)
            fig = px.bar(subset_df2, 
                         x='representation', 
                         y='value', 
                         color='embedding_type',                                                     # Color by embedding type
                         title=f'{measure} Performance Comparison on {dataset}',
                         labels={"value": "Performance Metric", "representation": "Representation"},
                         hover_data=['representation', 'value', 'embedding_type'],                    # Include representation, value, and embedding type in hover
                         category_orders={"representation": model_rep_order})                        # Explicit sorting order

            # Adjust layout to ensure proper alignment and equal spacing, and add legend at the top-right
            fig.update_layout(
                title={
                    'text': f'{dataset} {measure} model performance [by representation]',
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
                legend_title_text='Embedding Type',                                                  # Updated legend title to reflect embedding type
                legend=dict(
                    orientation="v",
                    y=1,
                    x=1,
                    xanchor='right',
                    yanchor='top'
                ),
                bargap=0.2,  # Increase spacing between bars
            )

            # Ensure the x-axis is treated as categorical and sorted, and rotate the labels
            fig.update_xaxes(
                title_text='Representation',
                type='category',                                                # Treat x-axis as categorical to prevent reordering
                tickangle=-45,                                                  # Rotate the x-axis labels
                tickfont=dict(size=9)                                           # Make the x-axis labels slightly smaller
            )

            fig.update_yaxes(
                title_text='Performance', 
                range=[y_axis_threshold, 1]
            )

            # Get the current date in YYYY-MM-DD format
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Save the plot in the specified output directory
            plot_file_name = f"{dataset}_{measure}_performance_comparison.{current_date}.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved plot for {measure} on dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()

    return df



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def gen_csvs_all(df, output_dir, neural=False, debug=False):
    """
    Generate CSV and HTML summary performance data for each dataset, combining all models into one file.

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_dir (str, required): Output directory for files
        neural (bool, optional): Whether the data is from deep learning models or classical ML models
        debug (bool, optional): Whether to print debug information
    """
    
    CSV_MEASURES = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy', 'te-recall', 'te-precision']  
    
    print("\n\tGenerating CSVs (all version)...")

    if debug:
        print("CSV DataFrame:\n", df)

    if not neural:
        df['M-Embeddings'] = df['embeddings'].apply(lambda x: x.split(':')[0])
        df['M-Mix'] = df['embeddings'].apply(lambda x: x.split(':')[1])
    else:
        df['M-Embeddings'] = df['embeddings']
        df['M-Mix'] = 'solo'  

    filtered_df = df[df['measure'].isin(CSV_MEASURES)]

    if debug:
        print("Filtered CSV DataFrame:\n", filtered_df)

    current_date = datetime.now().strftime("%Y-%m-%d")

    for dataset_tuple, group_df in filtered_df.groupby(['dataset']):
        dataset = dataset_tuple[0]                                              # dataset_tuple is a tuple, extract first element
        output_html = f"{output_dir}/{dataset}_results_{current_date}.html"
        output_csv = f"{output_dir}/{dataset}_results_{current_date}.csv"
        render_data_all(group_df, dataset, output_html, output_csv, debug)



def render_data_all(dataframe, dataset, output_html, output_csv, debug):
    """
    Render data for all models into a single file, either html or csv format
    
    Arguments:
    - dataframe: the input data
    - dataset: the dataset name
    - output_html: the output HTML file name
    - output_csv: the output CSV file name
    
    Returns:
    - None
    """
    print("\n\tRendering data (all version)...")

    if (debug):
        print(f"dataset: {dataset}, output_html: {output_html}, output_csv: {output_csv}, debug: {debug}")
        print("DataFrame:\n", dataframe)
        
    # Define measure order
    measure_order = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy', 'te-precision', 'te-recall']
    # Ensure dataframe measures are sorted by predefined order
    measure_category = pd.Categorical(dataframe['measure'], categories=measure_order, ordered=True)
    dataframe['measure'] = measure_category
    dataframe.sort_values(by=['measure'], inplace=True)

    grouped = dataframe.groupby(['M-Embeddings', 'M-Mix', 'class_type', 'model'], as_index=False)
    selected_columns = ['class_type', 'comp_method', 'M-Embeddings', 'M-Mix', 'representation', 'dimensions', 'measure', 'value', 'timelapse']

    rows = []
    csv_rows = [['Dataset', 'Model', 'Class Type', 'Comp Method', 'Embeddings', 'Mix', 'Representation', 'Dimensions', 'Measure', 'Value', 'Timelapse (Seconds)']]
    previous_embeddings = None
    previous_mix = None

    for (embeddings, mix, class_type, model), group in grouped:
        group_border = "border-top: 3px solid black;" if embeddings != previous_embeddings else ""
        first_row = True

        for _, row in group.iterrows():
            formatted_value = f"{row['value']:.3f}"
            formatted_timelapse = f"{row['timelapse']:,.0f}"

            if first_row:
                row_html = f"<tr style='font-size: 12px; {group_border}'><td><b>{row['M-Embeddings']}</b></td><td><i>{row['M-Mix']}</i></td>"
                first_row = False
            else:
                dotted_border = "border-bottom: 1px dotted gray;" if mix != previous_mix else ""
                row_html = f"<tr style='font-size: 12px; {dotted_border}'><td></td><td></td>"

            row_html += f"<td>{row['comp_method']}</td><td>{row['representation']}</td><td>{row['dimensions']}</td><td>{row['measure']}</td><td>{formatted_value}</td><td>{formatted_timelapse}</td></tr>"
            rows.append(row_html)

            csv_row = [dataset, model, row['class_type'], row['comp_method'], row['M-Embeddings'], row['M-Mix'], row['representation'], row['dimensions'], row['measure'], formatted_value, formatted_timelapse]
            csv_rows.append(csv_row)

        previous_embeddings = embeddings
        previous_mix = mix

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
    <tr><th>Embeddings</th><th>Mix</th><th>Comp Method</th><th>Representation</th><th>Dimensions</th><th>Measure</th><th>Value</th><th>Timelapse (Seconds)</th></tr>
    """
    table_html += "".join(rows)
    table_html += "</table>"

    if (debug):
        print("Dataset:", dataset)

    with open(output_html, 'w') as f:
        f.write(f"<h2>Results for Dataset: {dataset}</h2>")
        f.write(table_html)

    print(f"HTML Table saved as {output_html}.")

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)

    print(f"CSV saved as {output_csv}.")



def gen_csvs(df, output_dir, neural=False, debug=False):
    """
    generate CSV summary performance data for each data set, grouped by model 

    Args:
        df (Dataframe, required): input data, NB: the input data should be filtered for the measures of interest before calling this function
        output_dir (str, required): output directory of files
        neural (bool, optional): whether or not data is from the deep learning / neural models or classic ML models
        debug (bool, optional): whether or not to print out debug info.
    """
    
    print("\n\tgenerating CSVs...")

    if debug:
        print("CSV DataFrame:\n", df)
    
    if not neural:
        # split the 'embeddings' column into 'M-Embeddings' and 'M-Mix'
        df['M-Embeddings'] = df['embeddings'].apply(lambda x: x.split(':')[0])
        df['M-Mix'] = df['embeddings'].apply(lambda x: x.split(':')[1])        
    else:
        df['M-Embeddings'] = df['embeddings']
        df['M-Mix'] = 'solo'                        # default to solo for neural models

    # Filter the dataframe for the required measures
    filtered_df = df[df['measure'].isin(CSV_MEASURES)]

    if debug:
        print("Filtered CSV DataFrame:\n", filtered_df)

    # Get the current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Iterate through each dataset and model to generate tables
    for (dataset, model), group_df in filtered_df.groupby(['dataset', 'model']):
        output_html = f"{output_dir}/{dataset}_{model}_results.{current_date}.html"
        output_csv = f"{output_dir}/{dataset}_{model}_results.{current_date}.csv"
        # Assuming render_data is a function you have defined to output HTML and CSV
        render_data(group_df, dataset, model, output_html, output_csv)




def render_data(dataframe, dataset, model, output_html, output_csv):

    print("\n\trendering data...")

    print("dataframe:\n", dataframe)

    # Group the data by embeddings and mix (formerly Mode) within each embedding
    grouped = dataframe.groupby(['M-Embeddings', 'M-Mix', 'class_type'], as_index=False)
    
    # Select only the required columns, including the new 'dimensions' column
    selected_columns = ['class_type', 'comp_method', 'M-Embeddings', 'M-Mix', 'representation', 'dimensions', 'measure', 'value', 'timelapse']

    # Create an HTML table manually, ensuring that embeddings and mix only display once per group
    rows = []
    previous_embeddings = None
    previous_mix = None

    # Prepare CSV data
    csv_rows = [['dataset', 'class_type', 'comp_method', 'M-Embeddings', 'M-Mix', 'representation', 'dimensions', 'measure', 'value', 'timelapse (seconds)']]

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
                row_html = f"<tr style='font-size: 12px; {group_border}'><td><b>{row['M-Embeddings']}</b></td><td><i>{row['M-Mix']}</i></td>"
                first_row = False
            else:
                # Leave the embeddings and mix columns empty for subsequent rows, apply dotted line between Mix combinations
                dotted_border = "border-bottom: 1px dotted gray;" if mix != previous_mix else ""
                row_html = f"<tr style='font-size: 12px; {dotted_border}'><td></td><td></td>"

            # Add the rest of the columns (comp_method, representation, dimensions, measure, formatted value, formatted timelapse)
            row_html += f"<td>{row['comp_method']}</td><td>{row['representation']}</td><td>{row['dimensions']}</td><td>{row['measure']}</td><td>{formatted_value}</td><td>{formatted_timelapse}</td></tr>"
            rows.append(row_html)

            # Prepare the CSV row
            csv_row = [row['dataset'], row['class_type'], row['comp_method'], row['M-Embeddings'], row['M-Mix'], row['representation'], row['dimensions'], row['measure'], formatted_value, formatted_timelapse]
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

    print("dataset:", dataset)
    print("model:", model)
    #print("class_type:", class_type)

    # Write the HTML table to file, including class_type in the title
    with open(output_html, 'w') as f:
        #f.write(f"<h2>Results for Dataset: {dataset}, Model: {model}, Class Type: {class_type}</h2>")
        f.write(f"<h2>Results for Dataset: {dataset}, Model: {model}</h2>")
        f.write(table_html)

    print(f"HTML Table saved as {output_html}.")

    # Write CSV data to file
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)

    print(f"CSV saved as {output_csv}.")


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







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


def gen_summaries(df, output_path='../out', gen_file=True, stdout=False, debug=False):
    """
    Generate summaries for each dataset grouped by the first token in the embeddings type, writing to separate files for each dataset.
    """
    print(f'\n\tgenerating summary to {output_path}...')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter for Macro and Micro F1 scores only
    df_filtered = df[df['measure'].isin(['final-te-macro-f1', 'final-te-micro-f1'])]

    # Extract the first part of the embeddings as 'language_model'
    df_filtered['language_model'] = df_filtered['embeddings'].apply(lambda x: x.split(':')[0])

    # Get the current date in YYYY-MM-DD format for file naming
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Loop through each dataset and create a summary file
    for dataset in df_filtered['dataset'].unique():
        dataset_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        # Sort by necessary columns
        dataset_filtered.sort_values(by=['language_model', 'model', 'mode', 'representation', 'measure'], inplace=True)

        # Generate output file for each dataset
        file_name = f"{dataset}_summary_{current_date}.out"
        output_file = os.path.join(output_path, file_name)

        if gen_file:
            with open(output_file, 'w') as f:
                last_lang_model = None
                for lang_model, group in dataset_filtered.groupby('language_model'):
                    if last_lang_model is not None:
                        f.write("\n" + "-" * 80 + "\n")  # Separator line

                    # Formatting the table for the current language model group
                    formatted_table = tabulate(group, headers='keys', tablefmt='pretty', showindex=False)
                    f.write(f"Language Model: {lang_model}\n")
                    f.write(formatted_table)
                    f.write("\n")

                    last_lang_model = lang_model  # Update last language model encountered

            print(f"Output saved to {output_file}")

        if stdout:
            print(formatted_table)

# ----------------------------------------------------------------------------------------------------------------------------



def gen_summary_all(df, output_path='../out', gen_file=True, stdout=False, debug=False):
    """
    generate_summary_all()
    
    analyze the model performance results, print summary either to sdout or file
    """

    print(f'\n\tgenerating summary for all datasets to {output_path}...')

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter for Macro and Micro F1 scores only
    df_filtered = df[df['measure'].isin(MEASURES)]

    # Group data by 'dataset', 'model', 'embeddings', 'mode', 'representation', 'measure'
    result = df_filtered.groupby(['dataset', 'model', 'embeddings', 'mode', 'representation', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df_filtered, result, how='inner', on=['dataset', 'model', 'embeddings', 'mode', 'representation', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'model', 'embeddings', 'mode', 'representation', 'measure', 'value'])

    # Specify the column order
    columns_order = ['class_type', 'comp_method', 'model', 'dataset', 'embeddings', 'mode', 'representation', 'dimensions', 'measure', 'value', 'optimized', 'timelapse', 'run', 'epoch', 'os', 'cpus', 'gpus', 'mem']		

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in columns_order if col not in unique_result.columns]
    if missing_columns:
        print(f"Missing columns in DataFrame: {missing_columns}")
        return  

    # Sort the DataFrame
    final_result = unique_result[columns_order].copy()
    final_result.sort_values(by=['dataset', 'model', 'embeddings', 'mode', 'representation', 'measure'], inplace=True)

    if (debug):
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
    if (output_path and gen_file):
        print("generating file to output directory:", output_path)

        # Get the current date in YYYY-MM-DD format
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Add the date to the file name
        file_name = f"layercake.all.summary.{current_date}.out"
        output_file = os.path.join(output_path, file_name)
        
        # Write the output to the file
        with open(output_file, 'w') as f:
            f.write(final_formatted_table)
        
        print(f"Output saved to {output_file}")
    
    if (output_path and stdout):
        print(final_formatted_table)

# ----------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    print("\n\t----- Results Analysis -----")
    
    parser = argparse.ArgumentParser(description="Analyze model results and generate charts and/or summaries")

    parser.add_argument('file_path', type=str, help='Path to the TSV file with the data')
    parser.add_argument('-c', '--charts', action='store_true', default=False, help='Generate charts')
    parser.add_argument('-r', '--runtimes', action='store_true', default=False, help='Generate timrlapse charts')
    parser.add_argument('-n', '--neural', action='store_true', default=False, help='Output from Neural Nets')
    parser.add_argument('-s', '--summary', action='store_true', default=False, help='Generate summary')
    parser.add_argument('-o', '--output_dir', action='store_true', help='Directory to write output files. If not provided, defaults to ' + OUT_DIR + ' + the base file name of the input file.')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('-y', '--ystart', type=float, default=Y_AXIS_THRESHOLD, help='Y-axis starting value for the charts (default: 0.6)')
    parser.add_argument('-m', '--results', type=float, default=NUM_RESULTS, help='Y-axis starting value for the charts (default: 0.6)')
    parser.add_argument('-show', action='store_true', default=False, help='Display charts interactively (requires -c)')

    args = parser.parse_args()
    print("args: ", args)

    # Ensure at least one operation is specified
    if not (args.charts or args.summary):
        parser.error("No action requested, add -c for charts or -s for summary")

    debug = args.debug
    print("debug mode:", debug)

    print("y_start:", args.ystart)

    df = read_data_file(args.file_path, debug=debug)
    if (df is None):
        print("Error: Data file not found or empty")
        exit()

    out_dir = OUT_DIR
    if (debug):
        print("output directory:", out_dir)

    if (debug):
        print("Data file read successfully, df:", df.shape)

    # Create an output directory with today's date (format: YYYYMMDD)
    today_date = datetime.today().strftime('%Y%m%d')
    
    # Print just the filename without directories
    input_file = os.path.basename(args.file_path)
    print(f"Input file name: {input_file}")

    out_dir = os.path.join('../out/', today_date)
    out_dir = out_dir + '.' + input_file

    print("output directory:", out_dir)

    summ_dir = os.path.join(out_dir, 'summaries')
    timelapse_dir = os.path.join(out_dir, 'timelapse')
    charts_dir = os.path.join(out_dir, 'charts')
    
    # Check if output directory exists
    if not os.path.exists(out_dir):

        print("\n")

        create_dir = input(f"Output directory {out_dir} does not exist. Do you want to create it? (y/n): ").strip().lower()
        if create_dir == 'y':
            os.makedirs(out_dir)
            os.makedirs(summ_dir)
            os.makedirs(timelapse_dir)
            os.makedirs(charts_dir)
            print(f"Directory {out_dir} created, along with subdirectories")
        else:
            print(f"Directory {out_dir} was not created. Exiting.")
            exit()

    #
    # generate summaries
    #
    if args.summary:
        
        gen_summary_all(
            df, 
            summ_dir, 
            debug=debug
        )

        gen_summaries(
            df, 
            summ_dir, 
            gen_file=True, 
            stdout=False, 
            debug=debug
        )
        
        # gen_csvs(df, out_dir, neural=args.neural, debug=debug)
        gen_csvs_all(
            df, 
            summ_dir, 
            neural=args.neural, 
            debug=debug
        )
        
    #
    # generate charts
    #
    if args.charts:
        
        model_performance_comparison(
            df, 
            charts_dir, 
            neural=args.neural,
            show_charts=args.show, 
            y_axis_threshold=args.ystart,
            num_results=args.results,               # number of top results to display in the plot
            debug=debug
        )
        
        #
        # matplotlib option is less interactive but handles more test cases - its split by dataset 
        # and model as opposed to just dataset as the plotly graphs are designed for 
        #
        generate_charts_matplotlib(
            df, 
            charts_dir,
            neural=args.neural,
            show_charts=args.show,
            y_axis_threshold=args.ystart,
            debug=debug
        )

    #
    # generate timelapse plots
    #        
    if (args.runtimes):
        gen_timelapse_plots(
            df, 
            timelapse_dir, 
            show_charts=args.show,
            debug=debug
        )
    
