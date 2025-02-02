import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

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
MEASURES = ['final-te-macro-f1', 'final-te-micro-f1']

# Define the measures to be included in CSV file output
CSV_MEASURES = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy', 'te-recall', 'te-precision']  

Y_AXIS_THRESHOLD = 0.25                     # when to start the Y axis to show differentiation in the plot
TOP_N_RESULTS = 10                          # default number of results to display

#
# -----------------------------------------------------------------------------------------------------------------------------------



def generate_matplotlib_charts_by_model(
    df, 
    output_path='../out', 
    neural=False, 
    y_axis_threshold=Y_AXIS_THRESHOLD, 
    show_charts=False, 
    debug=False):
    """
    Generates per model bar charts for word-based, subword-based, and token-based models. Includes timelapse values on separate y axis
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be plotted.
        output_path (str): Directory to save the output files.
        neural (bool): Whether the data is from deep learning models or classical ML models.
        y_axis_threshold (float): The minimum value for the y-axis.
        show_charts (bool): Whether to display the charts interactively.
        debug (bool): Whether to print additional debug information during
        
    Returns:
        None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating combined charts for all language models...")

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
                    f"Dataset: {dataset}, Model: {model}, Measure: {measure} [by representation]",
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
                plot_file_name = f"{dataset}_{measure}_{model}.png"
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

    print("\n\tGenerating separate charts per model, dataset, and language model type...")

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
                    plot_file_name = f"{dataset}_{measure}_{model}_{embedding_category}.png"
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


def gen_timelapse_plots(
    df, 
    output_path='../out', 
    neural=False, 
    show_charts=False, 
    debug=False):
    """
    Generate timelapse plots for each dataset, model, embeddings, representation, and dimensions using Bokeh for interactive visualizations.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing timelapse data.
        output_path (str): Directory to save the output files.
        neural (bool): Whether the data is from deep learning models or classical ML models.
        show_charts (bool): Whether to display the charts interactively.
        debug (bool): Whether to enable debug mode for extra outputs.
    """
    print("\n\tGenerating timelapse plots...")

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_timelapse = df[['dataset', 'model', 'embeddings', 'representation', 'dimensions', 'timelapse']].drop_duplicates()

    if df_timelapse.empty:
        print("Error: No data available for timelapse analysis.")
        return

    for dataset in df_timelapse['dataset'].unique():
        
        if (debug):
            print(f"Generating timelapse plots for dataset {dataset}...")

        subset_df = df_timelapse[df_timelapse['dataset'] == dataset].copy()

        if subset_df.empty:
            print(f"No timelapse data available for dataset {dataset}")
            continue

        subset_df.sort_values(by='dimensions', ascending=False, inplace=True)
        subset_df['representation_with_dim'] = subset_df['representation'] + ' (' + subset_df['dimensions'].astype(str) + ')'
        
        if not neural:
            subset_df['embeddings_prefix'] = subset_df['embeddings'].apply(lambda x: x.split(':')[0])  # Extract prefix
        else:
            subset_df['embeddings_prefix'] = subset_df['embeddings']
            
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

        p.add_tools(HoverTool(tooltips=[("Classifier", "@classifier"), ("Embeddings", "@embeddings_prefix"), ("Timelapse", "@timelapse")]))

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


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def plotly_model_performance_horizontal(
    df, 
    output_path='../out', 
    neural=False, 
    y_axis_threshold=Y_AXIS_THRESHOLD, 
    num_results=TOP_N_RESULTS, 
    show_charts=True, 
    debug=False):
    """
    plotly_model_performance_horizontal() generates interactive, horizontal bar charts for each combination of model, dataset, and measure from a given DataFrame.
    
    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning models or classical ML models.
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - num_results (default: None): Number of top results to display in the plot (None for all results).
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during
    
    Returns:
    - None: The function saves the generated plots as HTML files in the specified output directory.
    """
    
    # Filter the dataframe for the measures of interest right away
    df = df[df['measure'].isin(MEASURES)]
    
    if df.empty:
        print("No data available for the specified measures")
        return

    print("\n\tgenerating plotly summary charts to output directory:", output_path)

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

        if (debug):
            print("processing dataset:", dataset)

        for measure in MEASURES:

            if (debug):
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
            subset_df2['representation'] = subset_df2.apply(lambda row: f"{row['representation']} - {row['dimensions']}", axis=1)

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
                legend_title_text='Language Model',                                                  # Updated legend title to reflect embedding type
                legend=dict(
                    orientation="v",
                    y=1,
                    x=1,
                    xanchor='right',
                    yanchor='top'
                ),
                bargap=0.15,            
                height=1000,            # Height of the chart
                width=1200              # Width of the chart
            )

            # Ensure the x-axis is treated as categorical and sorted, and rotate the labels
            fig.update_xaxes(
                title_text='Representation - Dimension',
                type='category',                                                # Treat x-axis as categorical to prevent reordering
                tickangle=-45,                                                  # Rotate the x-axis labels
                tickfont=dict(size=9)                                           # Make the x-axis labels slightly smaller
            )

            fig.update_yaxes(
                #title_text='Performance', 
                title_text=measure,
                range=[y_axis_threshold, 1]
            )

            # Get the current date in YYYY-MM-DD format
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Save the plot in the specified output directory
            plot_file_name = f"{dataset}_{measure}_performance_horizontal.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved plot for {measure} on dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()



def plotly_model_performance_dual_yaxis(
    df, 
    output_path='../out', 
    neural=False, 
    y_axis_threshold=0, 
    num_results=20, 
    show_charts=True, 
    debug=False):
    """
    Generates a dual-y-axis bar chart using Plotly for performance and timelapse values.
    The x-axis contains the representation labels, and the two y-axes show the measure value and timelapse values.

    Args:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path: Directory to save the output files.
    - neural: Whether the data is from deep learning models or classical ML models.
    - y_axis_threshold: The minimum value for the measure y-axis.
    - num_results: Number of top results to display.
    - show_charts: Whether to display the charts interactively.
    - debug: Whether to print debug information.

    Returns:
    - None: Saves the generated plots as HTML files.
    """
    # Filter for measures of interest
    df = df[df['measure'].isin(MEASURES)]
    if df.empty:
        print("No data available for the specified measures")
        return

    print("\n\tGenerating dual-y-axis charts with Plotly...")

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    if neural:
        df['embedding_type'] = df['embeddings']
    else:
        df['embedding_type'] = df['embeddings'].apply(lambda x: x.split(':')[0])

    for dataset in df['dataset'].unique():
        if debug:
            print("Processing dataset:", dataset)

        for measure in MEASURES:
            if debug:
                print(f"\n\tGenerating plots for dataset {dataset} with measure {measure}...")

            # Filter for the current dataset and measure
            subset_df = df[(df['dataset'] == dataset) & (df['measure'] == measure)]
            if subset_df.empty:
                print(f"No data available for dataset {dataset} with measure {measure}")
                continue

            subset_df = subset_df.groupby(['representation', 'model', 'dimensions', 'embeddings', 'embedding_type']).agg(
                {'value': 'max', 'timelapse': 'max'}
            ).reset_index()

            # Update representation labels to include dimensions
            subset_df['representation'] = subset_df.apply(lambda row: f"{row['representation']} - {row['dimensions']}", axis=1)

            # Sort by performance values
            subset_df.sort_values(by='value', ascending=False, inplace=True)

            # Limit to top results
            if num_results is not None:
                subset_df = subset_df.head(num_results)

            if debug:
                print("Subset after sorting:\n", subset_df)

            # Define the sorted order for the x-axis based on performance
            model_rep_order = subset_df['representation'].tolist()

            # Create the figure
            fig = go.Figure()

            # Add performance bars (color-coded by embedding type)
            for embedding_type in subset_df['embedding_type'].unique():
                embedding_data = subset_df[subset_df['embedding_type'] == embedding_type]
                fig.add_trace(go.Bar(
                    x=embedding_data['representation'],
                    y=embedding_data['value'],
                    name=embedding_type,
                    hovertemplate="<b>Representation:</b> %{x}<br><b>Performance:</b> %{y:.3f}<extra></extra>"
                ))

            # Add timelapse as scatter points (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=subset_df['representation'],
                y=subset_df['timelapse'],
                name='Timelapse',
                yaxis='y2',
                mode='markers',  # Only points, no lines
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate="<b>Representation:</b> %{x}<br><b>Timelapse:</b> %{y:.3f}<extra></extra>"
            ))

            # Update layout for dual y-axes
            fig.update_layout(
                title=dict(
                    text=f'Dataset: {dataset}, Measure: {measure} Model Performance with Timelapse',
                    x=0.5,
                    y=0.95,
                    font=dict(size=16, family='Arial', color='black', weight='bold')
                ),
                xaxis=dict(
                    title="Representation - Dimension",
                    title_font=dict(size=12, color='black', weight='bold'),
                    tickangle=-45,
                    tickfont=dict(size=9)
                ),
                yaxis=dict(
                    title=measure,
                    title_font=dict(size=12, color='blue', weight='bold'),
                    range=[y_axis_threshold, 1],
                    tickfont=dict(size=9)
                ),
                yaxis2=dict(
                    title="Timelapse (seconds)",
                    title_font=dict(size=12, color='red', weight='bold'),
                    overlaying='y',
                    side='right',
                    tickfont=dict(size=9, color='red')
                ),
                legend=dict(
                    title="Language Model",
                    orientation="v",
                    y=1,
                    x=1,
                    xanchor='right',
                    yanchor='top'
                ),
                bargap=0.15,            
                height=1000,            # Height of the chart
                width=1200              # Width of the chart
            )

            # Save the plot
            current_date = datetime.now().strftime("%Y-%m-%d")
            plot_file_name = f"{dataset}_{measure}_performance_with_timelapse.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)
            print(f"Saved plot for {measure} on dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()



def model_performance_comparison_all(df, output_path='../out', neural=False, y_axis_threshold=Y_AXIS_THRESHOLD, show_charts=True, debug=False):
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

            # Group by representation, classifier, dimensions, and embeddings to find maximum values
            subset_df2 = subset_df1.groupby(['representation', 'classifier', 'dimensions', 'embeddings', 'embedding_type']).agg({'value': 'max'}).reset_index()

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
            plot_file_name = f"{dataset}_{measure}_performance_comparison.html"
            plot_file = os.path.join(output_path, plot_file_name)
            fig.write_html(plot_file)

            print(f"Saved plot for {measure} on dataset {dataset} at {plot_file}")

            if show_charts:
                fig.show()

    return df



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def gen_csvs_all(df, output_dir, neural=False, debug=False):
    """
    Generate CSV and HTML summary performance data for each dataset, combining all classifiers into one file.

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_dir (str, required): Output directory for files
        neural (bool, optional): Whether the data is from deep learning classifiers or classical ML classifiers like SVM or Logistic Regression
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
    Render data for classifiers into a single file, either html or csv format
    
    Arguments:
    - dataframe: the input data
    - dataset: the dataset name
    - output_html: the output HTML file name
    - output_csv: the output CSV file name
    
    Returns:
    - None
    """
    if (debug):
        print("rendering data (all version)...")
        print(f"dataset: {dataset}, output_html: {output_html}, output_csv: {output_csv}, debug: {debug}")
        print("DataFrame:\n", dataframe)
        
    # Define measure order
    measure_order = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy', 'te-precision', 'te-recall']
    
    # Ensure dataframe measures are sorted by predefined order
    measure_category = pd.Categorical(dataframe['measure'], categories=measure_order, ordered=True)
    dataframe['measure'] = measure_category
    dataframe.sort_values(by=['measure'], inplace=True)

    grouped = dataframe.groupby(['M-Embeddings', 'M-Mix', 'class_type', 'classifier'], as_index=False)
    selected_columns = ['class_type', 'comp_method', 'M-Embeddings', 'M-Mix', 'representation', 'dimensions', 'measure', 'value', 'timelapse']

    rows = []
    csv_rows = [['Dataset', 'Classifier', 'Class Type', 'Comp Method', 'Embeddings', 'Mix', 'Representation', 'Dimensions', 'Measure', 'Value', 'Timelapse (Seconds)']]
    previous_embeddings = None
    previous_mix = None

    for (embeddings, mix, class_type, classifier), group in grouped:   
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

            csv_row = [dataset, classifier, row['class_type'], row['comp_method'], row['M-Embeddings'], row['M-Mix'], row['representation'], row['dimensions'], row['measure'], formatted_value, formatted_timelapse]
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
    generate CSV summary performance data for each data set, grouped by classifier 

    Args:
        df (Dataframe, required): input data, NB: the input data should be filtered for the measures of interest before calling this function
        output_dir (str, required): output directory of files
        neural (bool, optional): whether or not data is from the deep learning / neural classifiers or classic ML classifiers like SVM or LR
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
    for (dataset, classifier), group_df in filtered_df.groupby(['dataset', 'classifier']):
        output_html = f"{output_dir}/{dataset}_{classifier}_results.{current_date}.html"
        output_csv = f"{output_dir}/{dataset}_{classifier}_results.{current_date}.csv"
        # Assuming render_data is a function you have defined to output HTML and CSV
        render_data(group_df, dataset, classifier, output_html, output_csv)




def render_data(dataframe, dataset, classifier, output_html, output_csv):

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
    print("classifier:", classifier)
    #print("class_type:", class_type)

    # Write the HTML table to file, including class_type in the title
    with open(output_html, 'w') as f:
        f.write(f"<h2>Results for Dataset: {dataset}, Classifier: {classifier}</h2>")
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


def gen_dataset_summaries(
    df, 
    output_path='../out', 
    neural=False, 
    gen_file=True, 
    stdout=False, 
    debug=False):
    """
    Generate summaries for each dataset grouped by the first token in the embeddings type, writing to separate files for each dataset.
    
    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_path (str, optional): Output directory for files
        neural (bool, optional): Whether the data is from deep learning classifiers or classical ML classifiers
        gen_file (bool, optional): Whether to generate output files
        stdout (bool, optional): Whether to print output to stdout
        debug (bool, optional): Whether to print debug information
        
    Returns:
        None
    """
    print(f'\n\tgenerating summary to {output_path}...')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter for Macro and Micro F1 scores only
    df_filtered = df[df['measure'].isin(['final-te-macro-f1', 'final-te-micro-f1'])]

    if not neural:
        # Extract the first part of the embeddings as 'language_model'
        df_filtered['language_model'] = df_filtered['embeddings'].apply(lambda x: x.split(':')[0])
    else:
        df_filtered['language_model'] = df_filtered['embeddings']
           
    # Get the current date in YYYY-MM-DD format for file naming
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Loop through each dataset and create a summary file
    for dataset in df_filtered['dataset'].unique():
        dataset_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        # Sort by necessary columns
        dataset_filtered.sort_values(by=['language_model', 'classifier', 'mode', 'representation', 'measure'], inplace=True)

        # Generate output file for each dataset
        file_name = f"{dataset}_summary.out"
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
    
    analyze the classifier performance results, print summary either to stdout or file
    """

    print(f'\n\tgenerating summary for all datasets to {output_path}...')

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define measures of interest
    MEASURES = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy']

    # Filter for Macro and Micro F1 scores only
    df_filtered = df[df['measure'].isin(MEASURES)]

    # Group data by 'dataset', 'classifier', 'embeddings', 'mode', 'representation', 'measure'
    result = df_filtered.groupby(['dataset', 'classifier', 'embeddings', 'mode', 'representation', 'measure'])['value'].max().reset_index()

    # Merge the original data to fetch all corresponding column values
    merged_result = pd.merge(df_filtered, result, how='inner', on=['dataset', 'classifier', 'embeddings', 'mode', 'representation', 'measure', 'value'])
    unique_result = merged_result.drop_duplicates(subset=['dataset', 'classifier', 'embeddings', 'mode', 'representation', 'measure', 'value'])

    # Specify the column order
    columns_order = ['class_type', 'comp_method', 'classifier', 'dataset', 'embeddings', 'mode', 'representation', 'dimensions', 'measure', 'value', 'optimized', 'timelapse', 'run', 'epoch', 'os', 'cpus', 'gpus', 'mem']		

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in columns_order if col not in unique_result.columns]
    if missing_columns:
        print(f"Missing columns in DataFrame: {missing_columns}")
        return  

    # Sort the DataFrame
    final_result = unique_result[columns_order].copy()
    final_result.sort_values(by=['dataset', 'classifier', 'embeddings', 'mode', 'representation', 'measure'], inplace=True)

    if (debug):
        print("final result:\n", final_result)

    # Generate CSV file
    if gen_file:
        csv_file_name = f"layercake.all.summary.{datetime.now().strftime('%Y-%m-%d')}.csv"
        csv_output_file = os.path.join(output_path, csv_file_name)
        final_result.to_csv(csv_output_file, index=False)
        print(f"CSV summary saved to {csv_output_file}")

    # Generate formatted ASCII table
    formatted_table = tabulate(final_result, headers='keys', tablefmt='pretty', showindex=False)

    # Write ASCII output
    if gen_file:
        file_name = f"layercake.all.summary.{datetime.now().strftime('%Y-%m-%d')}.out"
        output_file = os.path.join(output_path, file_name)
        with open(output_file, 'w') as f:
            f.write(formatted_table)
        print(f"Output saved to {output_file}")

    if stdout:
        print(formatted_table)





def all_model_performance_time_horizontal(
    df, 
    output_path='../out', 
    neural=False, 
    y_axis_threshold=Y_AXIS_THRESHOLD, 
    top_n_results=TOP_N_RESULTS,
    show_charts=False, 
    debug=False):
    """
    Generates dataset, moedel representation performance bar charts for word-based, subword-based, and token-based models. Adds a 
    secondary y-axis for timelapse values.
    
    Args:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning classifiers or classical ML classisifers like SVM or LR.
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating combined charts for all language models...")

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
            
            # Filter data for the current combination
            df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset)].copy()

            if df_subset.empty:
                print(f"No data available for {measure}, in dataset {dataset}.")
                continue
            else:
                pass
                
            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

            # Combine representation and dimensions into a single label for x-axis
            df_subset['rep_dim'] = df_subset.apply(
                lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
            )
        
            # ---------------------------------------------------------------------------------------------
            # filter data depeneding upon what we are showing
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]                
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
            # ---------------------------------------------------------------------------------------------
            
            # Sort by measure value in descending order and limit to top N results
            df_subset = df_subset.sort_values(by='value', ascending=False).head(top_n_results)
            
            if df_subset.empty:
                print(f"No data available for {measure}, in dataset {dataset} after filtering.")
                continue

            if debug:
                print("df_subset:\n", df_subset.shape, df_subset)
                    
            # Create a color palette based on unique embedding types
            unique_embeddings = df_subset['embedding_type'].nunique()
            color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

            # Create the plot
            fig, ax1 = plt.subplots(figsize=(16, 10))

            # Primary y-axis for the measure values
            bars = sns.barplot(
                data=df_subset,
                x='rep_dim',
                y='value',
                hue='embedding_type',
                palette=color_palette,
                order=df_subset['rep_dim'],
                ax=ax1,
                orient='v',  # Vertical orientation
                dodge=False  # Ensures single bar per category
            )
            
            # Add metric values at the top of each bar
            for bar, value in zip(bars.patches, df_subset['value']):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,                                                  # x position
                    bar.get_height(),                                                                   # y position (height of the bar)
                    f"{value:.3f}",                                                                     # Text to display (rounded to 3 decimals)
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='normal'
                )

            # Customize the primary y-axis
            ax1.set_title(f"Dataset: {dataset}, Measure: {measure} [by representation]", fontsize=12, weight='bold')
            ax1.set_ylabel(measure, fontsize=10, weight='bold')
            ax1.set_ylim(y_axis_threshold, 1)
            ax1.tick_params(axis='y', labelsize=8)
            ax1.legend(title="Language Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=10)

            # Secondary y-axis for timelapse values
            ax2 = ax1.twinx()
            ax2.scatter(
                df_subset['rep_dim'],
                df_subset['timelapse'],
                color='black',
                marker='x',
                s=20,                      # size of the marker points
                label='Timelapse'
            )
            ax2.set_ylabel('Timelapse (seconds)', fontsize=10, color='red', weight='bold')
            ax2.tick_params(axis='y', labelsize=8, labelcolor='red')
            ax2.legend(loc='upper right', fontsize=10)

            # Adjust x-axis labels using set_xlabel
            ax1.set_xlabel("Representation:Dimensions", fontsize=10, weight='bold')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)

            # Adjust layout
            fig.tight_layout()

            # Save the plot
            plot_file_name = f"{dataset}_{measure}_horizontal.png"
            plt.savefig(os.path.join(output_path, plot_file_name), dpi=450)
            print(f"Saved plot to {output_path}/{plot_file_name}")

            # Optionally show the plot
            if show_charts:
                plt.show()





def model_performance_time_horizontal(
    df, 
    output_path='../out', 
    neural=False, 
    y_axis_threshold=Y_AXIS_THRESHOLD, 
    top_n_results=TOP_N_RESULTS,
    show_charts=False, 
    debug=False):
    """
    Generates dataset, moedel representation performance bar charts for word-based, subword-based, and token-based models. Adds a 
    secondary y-axis for timelapse values.
    
    Args:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning classifiers or classical ML classifiers.
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating combined charts for all language models...")

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
            for classifier in df['classifier'].unique():
                
                # Filter data for the current combination
                df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['classifier'] == classifier)].copy()

                if df_subset.empty:
                    print(f"No data available for {measure}, {classifier}, in dataset {dataset}.")
                    continue
                else:
                    pass
                    
                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

                # Combine representation and dimensions into a single label for x-axis
                df_subset['rep_dim'] = df_subset.apply(
                    lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                )
            
                # ---------------------------------------------------------------------------------------------
                # filter data depeneding upon what we are showing
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]                
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
                # ---------------------------------------------------------------------------------------------
                
                # Sort by measure value in descending order and limit to top N results
                df_subset = df_subset.sort_values(by='value', ascending=False).head(top_n_results)
                
                if df_subset.empty:
                    print(f"No data available for {measure}, {classifier}, in dataset {dataset} after filtering.")
                    continue

                if debug:
                    print("df_subset:\n", df_subset.shape, df_subset)
                        
                # Create a color palette based on unique embedding types
                unique_embeddings = df_subset['embedding_type'].nunique()
                color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

                # Create the plot
                fig, ax1 = plt.subplots(figsize=(16, 10))

                # Primary y-axis for the measure values
                bars = sns.barplot(
                    data=df_subset,
                    x='rep_dim',
                    y='value',
                    hue='embedding_type',
                    palette=color_palette,
                    order=df_subset['rep_dim'],
                    ax=ax1,
                    orient='v',                                 # Vertical orientation
                    dodge=False                                 # Ensures single bar per category
                )
                
                # Add metric values at the top of each bar
                for bar, value in zip(bars.patches, df_subset['value']):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,                                                  # x position
                        bar.get_height(),                                                                   # y position (height of the bar)
                        f"{value:.3f}",                                                                     # Text to display (rounded to 3 decimals)
                        ha='center', va='bottom', fontsize=8, color='black', fontweight='normal'
                    )

                # Customize the primary y-axis
                ax1.set_title(
                    f"Dataset:{dataset}, Classifier: {classifier}, Measure: {measure} [by representation]",
                    fontsize=18, weight='bold'
                )
                
                #ax1.set_ylabel(measure, fontsize=14)
                ax1.set_ylabel("Value", fontsize=14)
                ax1.set_ylim(y_axis_threshold, 1)
                ax1.tick_params(axis='y', labelsize=9)
                ax1.legend(title="Language Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)

                # Secondary y-axis for timelapse values
                ax2 = ax1.twinx()
                ax2.scatter(
                    df_subset['rep_dim'],
                    df_subset['timelapse'],
                    color='black',
                    marker='x',
                    s=30,                      # size of the marker points
                    label='Timelapse'
                )
                ax2.set_ylabel('Timelapse (seconds)', fontsize=14, color='red')
                ax2.tick_params(axis='y', labelsize=9, labelcolor='red')
                ax2.legend(loc='upper right', fontsize=12)

                # Adjust x-axis labels using set_xlabel
                ax1.set_xlabel("Representation:Dimensions", fontsize=14)
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)

                # Adjust layout
                fig.tight_layout()

                # Save the plot
                plot_file_name = f"{dataset}_{measure}_{classifier}_horizontal.png"
                plt.savefig(os.path.join(output_path, plot_file_name), dpi=450)
                print(f"Saved plot to {output_path}/{plot_file_name}")

                # Optionally show the plot
                if show_charts:
                    plt.show()



def model_performance_time_vertical(
    df,
    output_path='../out',
    neural=False,
    x_axis_threshold=0.0,
    top_n_results=TOP_N_RESULTS,
    show_charts=False,
    debug=False
):
    """
    Generates horizontal bar charts for word-based, subword-based, and token-based models on the same chart.
    Adds a secondary x-axis for timelapse values, and fits the output on a portrait layout.
    Filters to show only the top N results based on the metric value.
    
    Args:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning classifiers or classical ML classifiers like SVM or LR.
    - x_axis_threshold (default: 0.0): The minimum value for the x-axis.
    - top_n_results (default: 10): The number of top results to display.
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating horizontal charts for all language models...")

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
            for classifier in df['classifier'].unique():
                
                # Filter data for the current combination
                df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['classifier'] == classifier)].copy()

                if df_subset.empty:
                    print(f"No data available for {measure}, {classifier}, in dataset {dataset}.")
                    continue

                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

                # Combine representation and dimensions into a single label for y-axis
                df_subset['rep_dim'] = df_subset.apply(
                    lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                )

                # ---------------------------------------------------------------------------------------------
                # filter data to exclude specific embeddings
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
                # ---------------------------------------------------------------------------------------------

                # Sort by measure value in descending order and limit to top N results
                df_subset = df_subset.sort_values(by='value', ascending=False).head(top_n_results)

                if df_subset.empty:
                    print(f"No data available for {measure}, {classifier}, in dataset {dataset} after filtering.")
                    continue

                if debug:
                    print("df_subset:\n", df_subset.shape, df_subset)

                # Create a color palette based on unique embedding types
                unique_embeddings = df_subset['embedding_type'].nunique()
                color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

                # Create the plot
                fig, ax1 = plt.subplots(figsize=(14, 12))  # Increased width for better readability

                # Primary x-axis for the measure values
                bars = sns.barplot(
                    data=df_subset,
                    y='rep_dim',
                    x='value',
                    hue='embedding_type',
                    palette=color_palette,
                    order=df_subset['rep_dim'],
                    ax=ax1,
                    orient='h',                                     # Horizontal orientation
                    dodge=False                                     # Ensures single bar per category
                )
                
                # Add metric values at the end of each bar
                for bar, value in zip(bars.patches, df_subset['value']):
                    ax1.text(
                        bar.get_width() + 0.01,                                                     # x position (slightly beyond the end of the bar)
                        bar.get_y() + bar.get_height() / 2,                                         # y position (center of the bar)
                        f"{value:.3f}",                                                             # Text to display (rounded to 3 decimals)
                        ha='left', va='center', fontsize=8, color='black', fontweight='normal'
                    )

                # Customize the primary x-axis
                ax1.set_title(
                    f"Dataset: {dataset}, Classifier: {classifier}, Measure: {measure} [by representation]",
                    fontsize=14, weight='bold', ha='center'
                )
                ax1.set_xlabel(measure, fontsize=10, fontweight='normal')
                ax1.set_xlim(x_axis_threshold, 1)
                ax1.tick_params(axis='x', labelsize=9)
                ax1.legend(title="LM Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)

                # Secondary x-axis for timelapse values
                ax2 = ax1.twiny()
                ax2.scatter(
                    df_subset['timelapse'],
                    df_subset['rep_dim'],
                    color='black',
                    marker='x',
                    s=50,                           # size of the marker points
                    label='Timelapse'
                )
                ax2.set_xlabel('Timelapse (seconds)', fontsize=10, fontweight='normal', color='red')
                ax2.tick_params(axis='x', labelsize=9, labelcolor='red')
                ax2.legend(loc='upper right', fontsize=10)

                # Adjust y-axis labels
                ax1.set_ylabel("Representation:Dimensions", fontsize=10, fontweight='normal')
                ax1.tick_params(axis='y', labelsize=9)

                # Adjust layout
                fig.tight_layout()

                # Save the plot
                plot_file_name = f"{dataset}_{measure}_{classifier}_vertical.png"
                plt.savefig(os.path.join(output_path, plot_file_name), dpi=450)
                print(f"Saved plot to {output_path}/{plot_file_name}")

                # Optionally show the plot
                if show_charts:
                    plt.show()




def all_model_performance_time_vertical(
    df,
    output_path='../out',
    neural=False,
    x_axis_threshold=0.0,
    top_n_results=TOP_N_RESULTS,
    show_charts=False,
    debug=False
):
    """
    Generates horizontal bar charts for word-based, subword-based, and token-based language models for all classifier models on the same chart.
    Adds a secondary x-axis for timelapse values, and fits the output on a portrait layout. Filters to show only the top N results based on the metric value.
    
    Args:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
    - neural (default: False): Whether the data is from deep learning classifiers or classical ML classifiers like SVM or LR.
    - x_axis_threshold (default: 0.0): The minimum value for the x-axis.
    - top_n_results (default: 10): The number of top results to display.
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating horizontal charts for all language models...")

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

            # Filter data for the current combination
            df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset)].copy()

            if df_subset.empty:
                print(f"No data available for {measure} in dataset {dataset}.")
                continue

            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

            # Combine representation and dimensions into a single label for y-axis
            df_subset['rep_dim'] = df_subset.apply(
                lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
            )

            # ---------------------------------------------------------------------------------------------
            # filter data to exclude specific embeddings
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
            # ---------------------------------------------------------------------------------------------

            # Sort by measure value in descending order and limit to top N results
            df_subset = df_subset.sort_values(by='value', ascending=False).head(top_n_results)

            if df_subset.empty:
                print(f"No data available for {measure} in dataset {dataset} after filtering.")
                continue

            if debug:
                print("df_subset:\n", df_subset.shape, df_subset)

            # Create a color palette based on unique embedding types
            unique_embeddings = df_subset['embedding_type'].nunique()
            color_palette = sns.color_palette("colorblind", n_colors=unique_embeddings)

            # Create the plot
            fig, ax1 = plt.subplots(figsize=(14, 12))  # Increased width for better readability

            # Primary x-axis for the measure values
            bars = sns.barplot(
                data=df_subset,
                y='rep_dim',
                x='value',
                hue='embedding_type',
                palette=color_palette,
                order=df_subset['rep_dim'],
                ax=ax1,
                orient='h',                                 # Horizontal orientation
                dodge=False                                 # Ensures single bar per category
            )
            
            # Add metric values at the end of each bar
            for bar, value in zip(bars.patches, df_subset['value']):
                ax1.text(
                    bar.get_width() + 0.01,                                                     # x position (slightly beyond the end of the bar)
                    bar.get_y() + bar.get_height() / 2,                                         # y position (center of the bar)
                    f"{value:.3f}",                                                             # Text to display (rounded to 3 decimals)
                    ha='left', va='center', fontsize=8, color='black', fontweight='normal'
                )

            # Customize the primary x-axis
            ax1.set_title(
                f"Dataset: {dataset}, Measure: {measure} [by representation]",
                fontsize=12, weight='bold', ha='center'
            )
            ax1.set_xlabel(measure, fontsize=10, fontweight='bold')
            ax1.set_xlim(x_axis_threshold, 1)
            ax1.tick_params(axis='x', labelsize=8)
            ax1.legend(title="Language Model Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=10)

            # Secondary x-axis for timelapse values
            ax2 = ax1.twiny()
            ax2.scatter(
                df_subset['timelapse'],
                df_subset['rep_dim'],
                color='black',
                marker='x',
                s=20,                           # size of the marker points
                label='Timelapse'
            )
            ax2.set_xlabel('Timelapse (seconds)', fontsize=10, weight='bold', color='red')
            ax2.tick_params(axis='x', labelsize=8, labelcolor='red')
            ax2.legend(loc='upper right', fontsize=10)

            # Adjust y-axis labels
            ax1.set_ylabel("Representation:Dimensions", fontsize=10, fontweight='bold')
            ax1.tick_params(axis='y', labelsize=8)

            # Adjust layout
            fig.tight_layout()

            # Save the plot
            plot_file_name = f"{dataset}_{measure}_vertical.png"
            plt.savefig(os.path.join(output_path, plot_file_name), dpi=450)
            print(f"Saved plot to {output_path}/{plot_file_name}")

            # Optionally show the plot
            if show_charts:
                plt.show()



def generate_vertical_heatmap_by_model(
    df, 
    output_path='../out', 
    neural=False,
    top_n_results=None, 
    debug=False):
    """
    Generates a heatmap to display classifier performance for word-based, subword-based, and token-based embeddings.
    """
    print("\n\tGenerating heatmap for all language models...")

    # Filter for measures of interest
    df_measures = df[df['measure'].isin(MEASURES)]
    if debug:
        print("df:", df_measures.shape, df_measures)

    if df_measures.empty:
        print("Error: No data available for the specified measures.")
        return

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    for measure in MEASURES:
        for dataset in df['dataset'].unique():
            for classifier in df['classifier'].unique():
                
                # Filter and prepare data
                df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['classifier'] == classifier)].copy()
                if df_subset.empty:
                    print(f"No data available for {measure}, {classifier}, in dataset {dataset}.")
                    continue

                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

                # Combine representation and dimensions into a single label for y-axis
                df_subset['rep_dim'] = df_subset.apply(
                    lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                )

                # Pivot the data for heatmap
                heatmap_data = df_subset.pivot_table(
                    index='rep_dim', 
                    columns='embedding_type', 
                    values='value'
                )

                # Sort rows by the mean of metric values
                heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]

                if (top_n_results is not None) and (heatmap_data.shape[0] > top_n_results):
                    # Limit to top N results
                    heatmap_data = heatmap_data.head(top_n_results)

                # Plot heatmap
                plt.figure(figsize=(12, len(heatmap_data) * 0.5))
                sns.heatmap(
                    heatmap_data, 
                    annot=True, 
                    fmt=".3f", 
                    cmap="coolwarm", 
                    cbar_kws={"label": measure}, 
                    linewidths=0.5
                )
                plt.title(
                    f"Dataset: {dataset}, Measure: {measure}, Classifier: {classifier} [by representation]",
                    fontsize=12, weight='bold'
                )
                plt.xlabel("Language Model", fontsize=12, weight='bold')
                plt.ylabel("Representation:Dimensions", fontsize=12, weight='bold')
                                
                # Adjust the color bar label
                cbar = plt.gca().collections[0].colorbar
                cbar.ax.set_ylabel("Value", fontsize=12, weight='bold')  # Match the font size and bold weight
                
                plt.xticks(fontsize=8, rotation=45, ha="right")
                plt.yticks(fontsize=8)

                # Save the plot
                plot_file_name = f"{dataset}_{measure}_{classifier}_vertical_heatmap.png"
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, plot_file_name), dpi=300)
                print(f"Saved heatmap to {output_path}/{plot_file_name}")

                plt.close()



def generate_vertical_heatmap_all_models(
    df, 
    output_path='../out', 
    neural=False,
    top_n_results=None, 
    debug=False):
    """
    Generates a heatmap to display model performance for word-based, subword-based, and token-based embeddings.
    """

    print("\n\tGenerating heatmap for all language models and all classifiers...")

    # Filter for measures of interest
    df_measures = df[df['measure'].isin(MEASURES)]
    if debug:
        print("df:", df_measures.shape, df_measures)

    if df_measures.empty:
        print("Error: No data available for the specified measures.")
        return

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    for measure in MEASURES:
        for dataset in df['dataset'].unique():
            
            # Filter and prepare data
            df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset)].copy()
            if df_subset.empty:
                print(f"No data available for {measure}, in dataset {dataset}.")
                continue

            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

            # Combine representation and dimensions into a single label for y-axis
            df_subset['rep_dim'] = df_subset.apply(
                lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
            )

            # Pivot the data for heatmap
            heatmap_data = df_subset.pivot_table(
                index='rep_dim', 
                columns='embedding_type', 
                values='value'
            )

            # Sort rows by the mean of metric values
            heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]

            if (top_n_results is not None) and (heatmap_data.shape[0] > top_n_results):
                # Limit to top N results
                heatmap_data = heatmap_data.head(top_n_results)

            # Plot heatmap
            plt.figure(figsize=(12, len(heatmap_data) * 0.5))
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".3f", 
                cmap="coolwarm", 
                cbar_kws={"label": measure}, 
                linewidths=0.5
            )
            plt.title(
                f"Dataset: {dataset}, Measure: {measure} [by representation]",
                fontsize=12, weight='bold'
            )
            plt.xlabel("Language Model", fontsize=12, weight='bold')
            plt.ylabel("Representation:Dimensions", fontsize=12, weight='bold')
                            
            # Adjust the color bar label
            cbar = plt.gca().collections[0].colorbar
            cbar.ax.set_ylabel("Value", fontsize=12, weight='bold')  # Match the font size and bold weight
            
            plt.xticks(fontsize=8, rotation=45, ha="right")
            plt.yticks(fontsize=8)

            # Save the plot
            plot_file_name = f"{dataset}_{measure}_vertical_heatmap.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, plot_file_name), dpi=300)
            print(f"Saved heatmap to {output_path}/{plot_file_name}")

            plt.close()




def generate_horizontal_heatmap_by_model(
    df, 
    output_path='../out', 
    neural=False,
    top_n_results=None, 
    debug=False):
    """
    Generates a horizontal heatmap to display classifier performance for word-based, subword-based, and token-based language models.
    """
    print("\n\tGenerating horizontal heatmap for all language models...")

    # Filter for measures of interest
    df_measures = df[df['measure'].isin(MEASURES)]
    if debug:
        print("df:", df_measures.shape, df_measures)

    if df_measures.empty:
        print("Error: No data available for the specified measures.")
        return

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    for measure in MEASURES:
        for dataset in df['dataset'].unique():
            for classifier in df['classifier'].unique():
                
                # Filter and prepare data
                df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset) & (df['classifier'] == classifier)].copy()
                if df_subset.empty:
                    print(f"No data available for {measure}, {classifier}, in dataset {dataset}.")
                    continue

                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

                # Combine representation and dimensions into a single label for x-axis
                df_subset['rep_dim'] = df_subset.apply(
                    lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                )

                # Pivot the data for heatmap
                heatmap_data = df_subset.pivot_table(
                    index='embedding_type',  # Embedding types on y-axis
                    columns='rep_dim',       # Representations on x-axis
                    values='value'
                )

                # Sort columns by the mean of metric values
                heatmap_data = heatmap_data.loc[:, heatmap_data.mean(axis=0).sort_values(ascending=False).index]

                if (top_n_results is not None) and (heatmap_data.shape[1] > top_n_results):
                    # Limit to top N results
                    heatmap_data = heatmap_data.iloc[:, :top_n_results]

                # Plot heatmap
                plt.figure(figsize=(len(heatmap_data.columns), 16))
                sns.heatmap(
                    heatmap_data, 
                    annot=True, 
                    fmt=".3f", 
                    cmap="coolwarm", 
                    cbar_kws={"label": "Value"},  # Change the label from the measure to "Value"
                    linewidths=0.5
                )
                plt.title(
                    f"Dataset: {dataset}, Classifier: {classifier}, Measure: {measure} [by representation]",
                    fontsize=16, weight='bold'  # Increase font size and make bold
                )
                plt.xlabel("Representation:Dimensions", fontsize=12, weight='bold')             # Increase font size and bold
                plt.ylabel("Language Model", fontsize=12, weight='bold')                        # Increase font size and bold
                
                # Adjust the color bar label
                cbar = plt.gca().collections[0].colorbar
                cbar.ax.set_ylabel("Value", fontsize=12, weight='bold')  # Match the font size and bold weight
                
                plt.xticks(fontsize=10, rotation=45, ha="right")
                plt.yticks(fontsize=10)

                # Save the plot
                plot_file_name = f"{dataset}_{measure}_{classifier}_horizontal_heatmap.png"
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, plot_file_name), dpi=300)
                print(f"Saved heatmap to {output_path}/{plot_file_name}")

                plt.close()
                
                


def generate_horizontal_heatmap_all_models(
    df, 
    output_path='../out', 
    neural=False,
    top_n_results=None, 
    debug=False):
    """
    Generates a horizontal heatmap to display classifier performance for word-based, subword-based, and token-based language models.
    """
    
    print("\n\tGenerating horizontal heatmap for all languauge models and all classifiers...")

    # Filter for measures of interest
    df_measures = df[df['measure'].isin(MEASURES)]
    if debug:
        print("df:", df_measures.shape, df_measures)

    if df_measures.empty:
        print("Error: No data available for the specified measures.")
        return

    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    for measure in MEASURES:
        for dataset in df['dataset'].unique():
                
            # Filter and prepare data
            df_subset = df[(df['measure'] == measure) & (df['dataset'] == dataset)].copy()
            if df_subset.empty:
                print(f"No data available for {measure} in dataset {dataset}.")
                continue

            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])

            # Combine representation and dimensions into a single label for x-axis
            df_subset['rep_dim'] = df_subset.apply(
                lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
            )

            # Pivot the data for heatmap
            heatmap_data = df_subset.pivot_table(
                index='embedding_type',  # Embedding types on y-axis
                columns='rep_dim',       # Representations on x-axis
                values='value'
            )

            # Sort columns by the mean of metric values
            heatmap_data = heatmap_data.loc[:, heatmap_data.mean(axis=0).sort_values(ascending=False).index]

            if (top_n_results is not None) and (heatmap_data.shape[1] > top_n_results):
                # Limit to top N results
                heatmap_data = heatmap_data.iloc[:, :top_n_results]

            # Plot heatmap
            plt.figure(figsize=(len(heatmap_data.columns), 16))
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".3f", 
                cmap="coolwarm", 
                cbar_kws={"label": "Value"},  # Change the label from the measure to "Value"
                linewidths=0.5
            )
            plt.title(
                f"Dataset: {dataset}, Measure: {measure} [by representation]",
                fontsize=16, weight='bold'  # Increase font size and make bold
            )
            plt.xlabel("Representation:Dimensions", fontsize=12, weight='bold')             # Increase font size and bold
            plt.ylabel("Language Model", fontsize=12, weight='bold')                        # Increase font size and bold
            
            # Adjust the color bar label
            cbar = plt.gca().collections[0].colorbar
            cbar.ax.set_ylabel("Value", fontsize=12, weight='bold')  # Match the font size and bold weight
            
            plt.xticks(fontsize=10, rotation=45, ha="right")
            plt.yticks(fontsize=10)

            # Save the plot
            plot_file_name = f"{dataset}_{measure}_horizontal_heatmap.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, plot_file_name), dpi=300)
            print(f"Saved heatmap to {output_path}/{plot_file_name}")

            plt.close()



# ----------------------------------------------------------------------------------------------------------------------------

def analyze(
    df, 
    out_dir, 
    neural=False, 
    debug=False
):
    """
    Analyze performance by dataset and generate individual chart files for each measure.
    
    Args:
    - df: DataFrame containing the analysis data.
    - out_dir: Directory where the summary charts should be saved.
    - neural: Flag indicating neural or non-neural analysis.
    - debug: Debug mode flag.
    """

    print("\n\tGenerating performance analysis and charts by dataset...")

    # Extract language model and representation form
    if not neural:
        df[['language_model', 'representation_form']] = df['embeddings'].str.split(':', expand=True)
    else:
        df['language_model'] = df['embeddings']
        df['representation_form'] = 'solo'

    # Filter only supported measures
    supported_measures = MEASURES
    df_filtered = df[df['measure'].str.contains('|'.join(supported_measures))]

    # Group by dataset and generate charts for each
    datasets = df_filtered['dataset'].unique()
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Filter data for the specific dataset
        dataset_df = df_filtered[df_filtered['dataset'] == dataset]

        # Create a directory for this dataset
        dataset_dir = os.path.join(out_dir, f"{dataset}_charts")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Chart 1: Classifier Performance (one chart per measure)
        for measure in supported_measures:
            measure_df = dataset_df[dataset_df['measure'] == measure]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=measure_df, x='classifier', y='value', palette="Set2")
            plt.title(f"Summary Classifier Performance for Measure: {measure}")
            plt.xlabel("Classifier")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)

            classifier_chart = os.path.join(dataset_dir, f"{dataset}_classifier_performance_{measure}.png")
            plt.savefig(classifier_chart, bbox_inches='tight')
            plt.close()
            print(f"Saved: {classifier_chart}")

        # Chart 2: Performance by Language Model (one chart per measure)
        for measure in supported_measures:
            measure_df = dataset_df[dataset_df['measure'] == measure]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=measure_df, x='language_model', y='value', palette="pastel")
            plt.title(f"Performance by Language Model for Measure: {measure} - {dataset}")
            plt.xlabel("Language Model")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)

            language_model_chart = os.path.join(dataset_dir, f"{dataset}_language_model_performance_{measure}.png")
            plt.savefig(language_model_chart, bbox_inches='tight')
            plt.close()
            print(f"Saved: {language_model_chart}")

        # Chart 3: Performance by Representation Form (one chart per measure)
        for measure in supported_measures:
            measure_df = dataset_df[dataset_df['measure'] == measure]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=measure_df, x='representation_form', y='value', palette="muted")
            plt.title(f"Performance by Representation Form for Measure: {measure} - {dataset}")
            plt.xlabel("Representation Form")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)

            representation_form_chart = os.path.join(dataset_dir, f"{dataset}_representation_form_performance_{measure}.png")
            plt.savefig(representation_form_chart, bbox_inches='tight')
            plt.close()
            print(f"Saved: {representation_form_chart}")



# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n\t----- Layer Cake Analysis & Reporting -----")
    
    parser = argparse.ArgumentParser(description="Layer Cake Analysis and Reporting Engine")

    parser.add_argument('file_path', type=str, help='Path to the TSV file with the data')
    parser.add_argument('-c', '--charts', action='store_true', default=False, help='Generate charts')
    parser.add_argument('-m', '--heatmaps', action='store_true', default=False, help='Generate heatmaps')
    parser.add_argument('-t', '--runtimes', action='store_true', default=False, help='Generate timrlapse charts')
    parser.add_argument('-n', '--neural', action='store_true', default=False, help='Output from Neural Nets')
    parser.add_argument('-l', '--model', action='store_true', default=False, help='Generate model (classifier) specific charts')
    parser.add_argument('-s', '--summary', action='store_true', default=False, help='Generate summary')
    parser.add_argument('-o', '--output_dir', action='store_true', help='Directory to write output files. If not provided, defaults to ' + OUT_DIR + ' + the base file name of the input file.')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('-y', '--ystart', type=float, default=Y_AXIS_THRESHOLD, help='Y-axis starting value for the charts (default: 0.6)')
    parser.add_argument('-r', '--results', type=int, default=TOP_N_RESULTS, help=f'number of results to display (default: {TOP_N_RESULTS})')
    parser.add_argument('-show', action='store_true', default=False, help='Display charts interactively (requires -c)')
    parser.add_argument('-a', '--analyze', action='store_true', default=False, help='Generate analysis data')

    args = parser.parse_args()
    print("args: ", args)

    # Ensure at least one operation is specified
    if not (args.charts or args.summary or args.analyze):
        parser.error("No action requested, add -c for charts or -s for summary or -a for analyze")

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
    analysis_dir = os.path.join(out_dir, 'analysis')
    
    charts_model_dir = os.path.join(out_dir, 'model_charts')
    charts_summ_dir = os.path.join(out_dir, 'summary_charts')
    
    heatmap_model_dir = os.path.join(out_dir, 'model_heatmaps')
    heatmap_summ_dir = os.path.join(out_dir, 'summary_heatmaps')
    
    # Check if output directory exists
    if not os.path.exists(out_dir):

        print("\n")

        create_dir = input(f"Output directory {out_dir} does not exist. Do you want to create it? (y/n): ").strip().lower()
        
        if create_dir == 'y':
            os.makedirs(out_dir)
            os.makedirs(summ_dir)
            os.makedirs(timelapse_dir)
            os.makedirs(analysis_dir)
            
            os.makedirs(charts_model_dir)
            os.makedirs(charts_summ_dir)
            
            os.makedirs(heatmap_model_dir)
            os.makedirs(heatmap_summ_dir)
            
            print(f"Directory {out_dir} created, along with subdirectories")
        else:
            print(f"Directory {out_dir} was not created. Exiting.")
            exit()

    if (args.analyze):
        analyze(df, out_dir=analysis_dir, neural=args.neural, debug=args.debug)
        exit()
        
    #
    # generate charts
    #
    if args.charts:
        
        """
        #this generates matlibplots without timelapse data
        generate_matplotlib_charts_by_model(
            df=df, 
            output_path=charts_model_dir,
            neural=args.neural,
            y_axis_threshold=args.ystart,
            show_charts=args.show,
            debug=debug
        )
        """
        
        """
        #this generates matlibplots without timelapse data for eaxch embedding type
        generate_charts_matplotlib_split(
            df=df, 
            output_path=charts_model_dir,
            neural=args.neural,
            y_axis_threshold=args.ystart,
            show_charts=args.show,
            debug=debug
        )
        """
        
        # --------------------------------------------
        # across all models (classifiers)
        #
        
        plotly_model_performance_dual_yaxis(
                df=df, 
                output_path=charts_summ_dir, 
                neural=args.neural,
                y_axis_threshold=args.ystart,
                num_results=args.results,
                show_charts=args.show, 
                debug=debug
            )
         
        plotly_model_performance_horizontal(
                df=df, 
                output_path=charts_summ_dir, 
                neural=args.neural,
                y_axis_threshold=args.ystart,
                num_results=args.results,
                show_charts=args.show, 
                debug=debug
            )
        
        all_model_performance_time_horizontal(
            df=df, 
            output_path=charts_summ_dir,
            neural=args.neural,
            y_axis_threshold=args.ystart,
            top_n_results=args.results,
            show_charts=args.show,
            debug=debug
        )
        
        all_model_performance_time_vertical(
            df=df, 
            output_path=charts_summ_dir,
            neural=args.neural,
            x_axis_threshold=args.ystart,
            top_n_results=args.results,
            show_charts=args.show,
            debug=debug
        )
        
        #
        # ---------------------------------------------
        
        
        # --------------------------------------------
        # by model (classifier)
        #
        if args.model:
            
            model_performance_time_horizontal(
                df=df, 
                output_path=charts_model_dir,
                neural=args.neural,
                y_axis_threshold=args.ystart,
                top_n_results=args.results,
                show_charts=args.show,
                debug=debug
            )
            
            model_performance_time_vertical(
                df=df, 
                output_path=charts_model_dir,
                neural=args.neural,
                x_axis_threshold=args.ystart,
                top_n_results=args.results,
                show_charts=args.show,
                debug=debug
            )
        #
        # ---------------------------------------------
    
    
    #
    # generate heatmaps
    #
    if args.heatmaps:

        generate_vertical_heatmap_by_model(
            df, 
            output_path=heatmap_model_dir,
            neural=args.neural,
            top_n_results=args.results,
            debug=debug
        )

        generate_vertical_heatmap_all_models(
            df, 
            output_path=heatmap_summ_dir,
            neural=args.neural,
            top_n_results=args.results,
            debug=debug
        )

        
        generate_horizontal_heatmap_by_model(
            df, 
            output_path=heatmap_model_dir,
            neural=args.neural,
            top_n_results=args.results,
            debug=debug
        )

        generate_horizontal_heatmap_all_models(
            df, 
            output_path=heatmap_summ_dir,
            neural=args.neural,
            top_n_results=args.results,
            debug=debug
        )
        
            
    #
    # generate summaries
    #
    if args.summary:
        
        gen_summary_all(
            df=df, 
            output_path=summ_dir, 
            gen_file=True,
            debug=debug
        )

        gen_dataset_summaries(
            df=df, 
            output_path=summ_dir, 
            neural=args.neural,
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
    # generate timelapse plots
    #        
    if (args.runtimes):
        
        gen_timelapse_plots(
            df=df, 
            output_path=timelapse_dir, 
            neural=args.neural,
            show_charts=args.show,
            debug=debug
        )
    
