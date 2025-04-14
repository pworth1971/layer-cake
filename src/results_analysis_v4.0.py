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
import matplotlib.patches as mpatches

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
TOP_N_RESULTS = 25                          # default number of results to display

ML_CLASSIFIERS = ['svm', 'lr', 'nb']
DL_CLASSIFIERS = ['cnn', 'lstm', 'attn', 'hf.sc', 'hf.cnn']
#
# -----------------------------------------------------------------------------------------------------------------------------------



def plotly_model_performance_horizontal(
    df, 
    output_path='../out', 
    y_axis_threshold=Y_AXIS_THRESHOLD, 
    num_results=TOP_N_RESULTS, 
    show_charts=True, 
    debug=False):
    """
    plotly_model_performance_horizontal() generates interactive, horizontal bar charts for each combination of model, dataset, and measure from a given DataFrame.
    
    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
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

    # parse the embeddings field if its an ML model entry
    df['embedding_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

    """
    if neural:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings']
    else:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings'].apply(lambda x: x.split(':')[0])
    """

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
    y_axis_threshold=0, 
    num_results=TOP_N_RESULTS, 
    show_charts=True, 
    debug=False):
    """
    Generates a dual-y-axis bar chart using Plotly for performance and timelapse values.
    The x-axis contains the representation labels, and the two y-axes show the measure value and timelapse values.

    Args:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path: Directory to save the output files.
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

    # Extract language model and representation form based on classifier type
    df['embedding_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    
    """
    if neural:
        df['embedding_type'] = df['embeddings']
    else:
        df['embedding_type'] = df['embeddings'].apply(lambda x: x.split(':')[0])
    """

    for dataset in df['dataset'].unique():
        
        print("processing dataset:", dataset)

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

            # Filter to the top num_results          
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



def model_performance_comparison_all(df, output_path='../out', y_axis_threshold=Y_AXIS_THRESHOLD, show_charts=True, debug=False):
    """
    model_performance_comparison generates bar charts for each combination of model, dataset, and measure from a given DataFrame.
    
    Arguments:
    - df: Pandas DataFrame that contains the data to be plotted.
    - output_path (default: '../out'): Directory to save the output files. If it doesn't exist, it is created.
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

    # Extract language model and representation form based on classifier type
    df['embedding_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    
    """
    if neural:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings']
    else:
        # Extract the first term in the colon-delimited 'embeddings' field
        df['embedding_type'] = df['embeddings'].apply(lambda x: x.split(':')[0])
    """

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


def gen_csvs_all(df, chart_output_dir, csv_output_dir, debug=False):
    """
    Generate CSV and HTML summary performance data for each dataset, combining all classifiers into one file.

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_dir (str, required): Output directory for files
        debug (bool, optional): Whether to print debug information
    """
    
    CSV_MEASURES = ['final-te-macro-f1', 'final-te-micro-f1', 'te-accuracy', 'te-recall', 'te-precision']  
    
    print("\n\tGenerating CSVs (all version)...")

    if debug:
        print("CSV DataFrame:\n", df)

    df['M-Embeddings'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['M-Mix'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    """
    if not neural:
        df['M-Embeddings'] = df['embeddings'].apply(lambda x: x.split(':')[0])
        df['M-Mix'] = df['embeddings'].apply(lambda x: x.split(':')[1])
    else:
        df['M-Embeddings'] = df['embeddings']
        df['M-Mix'] = 'solo'  
    """

    filtered_df = df[df['measure'].isin(CSV_MEASURES)]

    if debug:
        print("Filtered CSV DataFrame:\n", filtered_df)

    current_date = datetime.now().strftime("%Y-%m-%d")

    for dataset_tuple, group_df in filtered_df.groupby(['dataset']):
        dataset = dataset_tuple[0]                                              # dataset_tuple is a tuple, extract first element
        output_html = f"{chart_output_dir}/{dataset}_results_{current_date}.html"
        output_csv = f"{csv_output_dir}/{dataset}_results_{current_date}.csv"
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



def gen_csvs(df, output_dir, debug=False):
    """
    generate CSV summary performance data for each data set, grouped by classifier 

    Args:
        df (Dataframe, required): input data, NB: the input data should be filtered for the measures of interest before calling this function
        output_dir (str, required): output directory of files
        debug (bool, optional): whether or not to print out debug info.
    """
    
    print("\n\tgenerating CSVs...")

    if debug:
        print("CSV DataFrame:\n", df)
    
    df['M-Embeddings'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['M-Mix'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    """
    if not neural:
        # split the 'embeddings' column into 'M-Embeddings' and 'M-Mix'
        df['M-Embeddings'] = df['embeddings'].apply(lambda x: x.split(':')[0])
        df['M-Mix'] = df['embeddings'].apply(lambda x: x.split(':')[1])        
    else:
        df['M-Embeddings'] = df['embeddings']
        df['M-Mix'] = 'solo'                        # default to solo for neural models
    """

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
    gen_file=True, 
    stdout=False, 
    debug=False):
    """
    Generate summaries for each dataset grouped by the first token in the embeddings type, writing to separate files for each dataset.
    
    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_path (str, optional): Output directory for files
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

    df_filtered['language_model'] = df_filtered.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

    """
    if not neural:
        # Extract the first part of the embeddings as 'language_model'
        df_filtered['language_model'] = df_filtered['embeddings'].apply(lambda x: x.split(':')[0])
    else:
        df_filtered['language_model'] = df_filtered['embeddings']
    """

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
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - top_n_results (default: 10): The number of top results to display in the plot.
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating horizontal combined charts for all language models...")

    if (debug):
        print(f"output_path: {output_path}, y_axis_threshold: {y_axis_threshold}, top_n_results: {top_n_results}, show_charts: {show_charts}")

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
            
            df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

            """
            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
            """

            # Combine representation and dimensions into a single label for x-axis
            df_subset['rep_dim'] = df_subset.apply(
                lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
            )
        
            # ---------------------------------------------------------------------------------------------
            # filter data depeneding upon what we are showing
            """
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]                
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
            """
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
    - y_axis_threshold (default: 0): The minimum value for the y-axis (used to set a threshold).
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating horizontal combined charts for all language models...")

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
                
                df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

                """
                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
                """

                # Combine representation and dimensions into a single label for x-axis
                df_subset['rep_dim'] = df_subset.apply(
                    lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
                )
            
                # ---------------------------------------------------------------------------------------------
                # filter data depeneding upon what we are showing
                """
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]                
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
                df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
                """
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
    - x_axis_threshold (default: 0.0): The minimum value for the x-axis.
    - top_n_results (default: 10): The number of top results to display.
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating vertical charts for all language models...")

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

                df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

                """
                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
                """

                # Combine representation and dimensions into a single label for y-axis
                df_subset['rep_dim'] = df_subset.apply(lambda row: f"{row['representation']}:{row['dimensions']}", axis=1)
                
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
    - x_axis_threshold (default: 0.0): The minimum value for the x-axis.
    - top_n_results (default: 10): The number of top results to display.
    - show_charts (default: True): Whether to display the charts interactively.
    - debug (default: False): Whether to print additional debug information during processing.
    
    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """
    print("\n\tGenerating vertical charts for all language models...")

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

            df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

            """
            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
            """

            # Combine representation and dimensions into a single label for y-axis
            df_subset['rep_dim'] = df_subset.apply(
                lambda row: f"{row['representation']}:{row['dimensions']}", axis=1
            )

            # ---------------------------------------------------------------------------------------------
            # filter data to exclude specific embeddings
            """
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('weighted', case=False, na=False)]
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('wce', case=False, na=False)]
            df_subset = df_subset[~df_subset['rep_dim'].str.contains('tce', case=False, na=False)]
            """
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
    top_n_results=None, 
    debug=False):
    """
    Generates a heatmap to display classifier performance for word-based, subword-based, and token-based embeddings.
    """
    print("\n\tGenerating vertical heatmap for all language models...")

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

                df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

                """
                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
                """

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
    top_n_results=None, 
    debug=False):
    """
    Generates a heatmap to display model performance for word-based, subword-based, and token-based embeddings.
    """

    print("\n\tGenerating vertical heatmap for all language models and all classifiers...")

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

            df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

            """
            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
            """

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

                df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

                """
                # Extract the base embeddings (everything before the colon)
                if neural:
                    df_subset['embedding_type'] = df_subset['embeddings']
                else:
                    df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
                """

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

            df_subset['embedding_type'] = df_subset.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)

            """
            # Extract the base embeddings (everything before the colon)
            if neural:
                df_subset['embedding_type'] = df_subset['embeddings']
            else:
                df_subset['embedding_type'] = df_subset['embeddings'].apply(lambda x: x.split(':')[0])
            """

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
#
# Global performance analysis charts (box plots)
#
#

def performance_analysis_detail(
    df, 
    out_dir, 
    debug=False, 
    include_opt=False
):
    """
    Analyze performance by dataset and generate individual chart files,
    including timelapse performance box plots for all performance categories.
    
    Args:
    - df: DataFrame containing the analysis data.
    - out_dir: Directory where the summary charts should be saved.
    - debug: Debug mode flag.
    - include_opt: Flag to include 'optimized' values in the timelapse boxplots.
    """

    print("\n\tGenerating performance analysis charts by Classifier, Language Model and Representation...")

    # Filter only supported measures
    supported_measures = MEASURES
    df_filtered = df[df['measure'].str.contains('|'.join(supported_measures))]

    # Extract language model and representation form based on classifier type
    df_filtered['language_model'] = df_filtered.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df_filtered['representation_form'] = df_filtered.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Apply the 'include_opt' filter for timelapse boxplots
    if not include_opt:
        df = df[df['optimized'] == False]

    # Group by dataset and generate charts for each
    datasets = df_filtered['dataset'].unique()
    for dataset in datasets:

        print(f"processing dataset: {dataset}")

        # Filter data for the specific dataset
        dataset_df = df_filtered[df_filtered['dataset'] == dataset]

        # --- CHART 1: Classifier Performance (one chart per measure) ---
        for measure in supported_measures:
            
            measure_df = dataset_df[dataset_df['measure'] == measure]

            # Plot classifier performance
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=measure_df, x='classifier', y='value', palette="Set2")
            plt.title(f"Classifier Performance Summary - dataset: {dataset}, measure: {measure}")
            plt.xlabel("Classifier")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)

            classifier_chart = os.path.join(out_dir, f"{dataset}_classifier_performance_{measure}.png")
            plt.savefig(classifier_chart, bbox_inches='tight')
            plt.close()
            print(f"Saved: {classifier_chart}")

        # --- CHART 2: Performance by Language Model (one chart per measure) ---
        for measure in supported_measures:
            measure_df = dataset_df[dataset_df['measure'] == measure]

            # Plot language model performance
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=measure_df, x='language_model', y='value', palette="pastel")
            plt.title(f"Language Model Performance Summary - dataset: {dataset}, measure: {measure}")
            plt.xlabel("Language Model")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)

            language_model_chart = os.path.join(out_dir, f"{dataset}_language_model_performance_{measure}.png")
            plt.savefig(language_model_chart, bbox_inches='tight')
            plt.close()
            print(f"Saved: {language_model_chart}")

        # --- CHART 3: Performance by Representation Form (one chart per measure) ---
        for measure in supported_measures:
            measure_df = dataset_df[dataset_df['measure'] == measure]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=measure_df, x='representation_form', y='value', palette="muted")
            plt.title(f"Representation Performance Summary - dataset: {dataset}, measure: {measure}")
            plt.xlabel("Representation Form")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)

            representation_form_chart = os.path.join(out_dir, f"{dataset}_representation_form_performance_{measure}.png")
            plt.savefig(representation_form_chart, bbox_inches='tight')
            plt.close()
            print(f"Saved: {representation_form_chart}")

        # --- TIMELAPSE CHART 1: Timelapse by Classifier ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dataset_df, x='classifier', y='timelapse', palette="Set2")
        plt.title(f"Timelapse Performance by Classifier - {dataset}")
        plt.xlabel("Classifier")
        plt.ylabel("Time Elapsed (seconds)")
        plt.xticks(rotation=45)

        timelapse_classifier_chart = os.path.join(out_dir, f"{dataset}_timelapse_by_classifier.png")
        plt.savefig(timelapse_classifier_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {timelapse_classifier_chart}")

        # --- TIMELAPSE CHART 2: Timelapse by Language Model ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dataset_df, x='language_model', y='timelapse', palette="pastel")
        plt.title(f"Timelapse Performance by Language Model - {dataset}")
        plt.xlabel("Language Model")
        plt.ylabel("Time Elapsed (seconds)")
        plt.xticks(rotation=45)

        timelapse_language_model_chart = os.path.join(out_dir, f"{dataset}_timelapse_by_language_model.png")
        plt.savefig(timelapse_language_model_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {timelapse_language_model_chart}")

        # --- TIMELAPSE CHART 3: Timelapse by Representation Form ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dataset_df, x='representation_form', y='timelapse', palette="muted")
        plt.title(f"Timelapse Performance by Representation Form - {dataset}")
        plt.xlabel("Representation Form")
        plt.ylabel("Time Elapsed (seconds)")
        plt.xticks(rotation=45)

        timelapse_representation_form_chart = os.path.join(out_dir, f"{dataset}_timelapse_by_representation_form.png")
        plt.savefig(timelapse_representation_form_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {timelapse_representation_form_chart}")



def perforamance_analysis_summary(df, out_dir, debug=False):
    """
    Analyze classifier and language model performance across all datasets.

    Args:
    - df: DataFrame containing the analysis data.
    - out_dir: Directory where the summary charts should be saved.
    - debug: Debug mode flag.

    Returns:
    - None: The function saves the generated plots as PNG files in the specified output directory.
    """

    print("\n\tAnalyzing classifier and language model performance in aggregate across all datasets...")
    
    # Create output directory if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")
    
    for measure in MEASURES:

        measure_df = df[df['measure'] == measure]

        #
        # if we are looking at ML data we need to parse the language model type from
        # the embeddings column in the log data, first value before colon ':'
        #
        # Extract language model and representation form based on classifier type
        measure_df['lm_type'] = measure_df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
        #x_axis = measure_df.apply(lambda row: 'lm_type' if row['classifier'] in ML_CLASSIFIERS else 'embeddings', axis=1).iloc[0]
        x_axis = 'lm_type'

        # --- CHART: Classifier performance across datasets and models ---
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=measure_df, x='classifier', y='value', palette="colorblind")
        plt.title(f"Classifier Performance Summary ({measure})", fontweight='bold')
        plt.xlabel("Classifier")
        plt.ylabel("Metric Value")
        plt.xticks(rotation=45, fontstyle='italic')

        classifier_chart = os.path.join(out_dir, f"classifier_performance_{measure}.png")
        plt.savefig(classifier_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {classifier_chart}")

        # --- CHART: Dual-axis chart combining performance and timelapse ---
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Primary Y-axis (Boxplot for Performance Metric)
        sns.boxplot(data=measure_df, x='classifier', y='value', ax=ax1, palette="colorblind")
        ax1.set_ylabel(f"{measure} Value", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Secondary Y-axis (Timelapse Scatter Plot)
        ax2 = ax1.twinx()
        sns.stripplot(data=measure_df, x='classifier', y='timelapse', ax=ax2, color='darkred', jitter=True, alpha=0.6, size=6)
        ax2.set_ylabel("Timelapse (seconds)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"Global Classifier Performance ({measure})")
        plt.xticks(rotation=45, fontstyle='italic')

        dual_axis_chart = os.path.join(out_dir, f"classifier_performance_timelapse_{measure}.png")
        plt.savefig(dual_axis_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {dual_axis_chart}")

        # --- CHART: Language model performance across classifiers and datasets ---
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=measure_df, x=x_axis, y='value', palette="colorblind")
        plt.title(f"Language Model Performance Summary ({measure})", fontweight='bold')
        plt.xlabel("Language Model Type")
        plt.ylabel("Metric Value")
        plt.xticks(rotation=45, fontstyle='italic')

        language_model_chart = os.path.join(out_dir, f"language_model_performance_{measure}.png")
        plt.savefig(language_model_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {language_model_chart}")

        # --- CHART: Dual-axis chart combining performance and timelapse ---
        fig, ax1 = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=measure_df, x=x_axis, y='value', ax=ax1, palette="colorblind")
        ax1.set_ylabel(f"{measure} Value", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        sns.stripplot(data=measure_df, x=x_axis, y='timelapse', ax=ax2, color='darkred', jitter=True, alpha=0.6, size=6)
        ax2.set_ylabel("Timelapse (seconds)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"Global Language Model Performance ({measure})")
        plt.xlabel("Language Model Type")
        plt.xticks(rotation=45, fontstyle='italic')

        dual_axis_chart = os.path.join(out_dir, f"language_model_performance_timelapse_{measure}.png")
        plt.savefig(dual_axis_chart, bbox_inches='tight')
        plt.close()
        print(f"Saved: {dual_axis_chart}")


# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


def gen_global_dataset_summaries(df, output_path=OUT_DIR):
    """
    Generate separate summary data files for each measure in MEASURES by dataset, 
    including both performance measures and execution time (timelapse).

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_path (str, optional): Output directory for files
    
    Returns:
        None
    """
    print(f'\n\tGenerating global dataset summary data to {output_path}...')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df['lm_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['representation'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    for measure in MEASURES:
        df_filtered = df[df['measure'] == measure]
        
        summary_df = df_filtered.pivot_table(
            index=["dataset", "classifier", "lm_type", "representation"],
            values=["value", "timelapse"],
            aggfunc=["mean", "max", "min"]
        )
        
        summary_df.reset_index(inplace=True)
        
        file_name = f"global_dataset_summary_{measure}.csv"
        output_file = os.path.join(output_path, file_name)
        summary_df.to_csv(output_file, index=False)
        print(f"Saved global dataset summary data to {output_file}")

        
def gen_global_classifier_summaries(df, output_path=OUT_DIR):
    """
    Generate separate summary data files for each measure in MEASURES by classifier, 
    including both performance measures and execution time (timelapse).

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_path (str, optional): Output directory for files
    
    Returns:
        None
    """
    print(f'\n\tGenerating global classifier summary data to {output_path}...')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df['lm_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['representation'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    for measure in MEASURES:
        df_filtered = df[df['measure'] == measure]
        
        summary_df = df_filtered.pivot_table(
            index=["classifier", "dataset", "lm_type", "representation"],
            values=["value", "timelapse"],
            aggfunc=["mean", "max", "min"]
        )
        
        summary_df.reset_index(inplace=True)
        
        file_name = f"global_classifier_summary_{measure}.csv"
        output_file = os.path.join(output_path, file_name)
        summary_df.to_csv(output_file, index=False)
        print(f"Saved global classifier summary data to {output_file}")

        
def gen_global_lm_summaries(df, output_path=OUT_DIR):
    """
    Generate separate summary data files for each measure in MEASURES by language model, 
    including both performance measures and execution time (timelapse).

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_path (str, optional): Output directory for files
    
    Returns:
        None
    """
    print(f'\n\tGenerating global language model summary data to {output_path}...')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df['lm_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['representation'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    for measure in MEASURES:
        df_filtered = df[df['measure'] == measure]
        
        summary_df = df_filtered.pivot_table(
            index=["lm_type", "representation", "dataset", "classifier"],
            values=["value", "timelapse"],
            aggfunc=["mean", "max", "min"]
        )
        
        summary_df.reset_index(inplace=True)
        
        file_name = f"global_lm_summary_{measure}.csv"
        output_file = os.path.join(output_path, file_name)
        summary_df.to_csv(output_file, index=False)
        print(f"Saved global language model summary data to {output_file}")


def gen_global_dataset_max_data(df, output_path=OUT_DIR):
    """
    Generate separate summary data files for each measure in MEASURES by language model, 
    including only the max performance measure values and their corresponding timelapse values.

    Args:
        df (DataFrame, required): Input data, already filtered for the measures of interest
        output_path (str, optional): Output directory for files
    
    Returns:
        None
    """
    print(f'\n\tgenerating global dataset max summary data to {output_path}...')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df['lm_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['representation'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    group_list = ["dataset", "classifier", "lm_type", "representation"]
    
    for measure in MEASURES:
        df_filtered = df[df['measure'] == measure]
        
        # Compute max summary statistics for the measure
        value_summary_df = df_filtered.pivot_table(
            index=group_list,
            values="value",
            aggfunc="max"
        )
        
        # Extract timelapse values corresponding to max measure values
        max_timelapse = df_filtered.loc[
            df_filtered.groupby(group_list)['value'].idxmax(), group_list + ['timelapse']
        ]
        #max_timelapse.rename(columns={'timelapse': 'timelapse_max'}, inplace=True)

        # Merge max value summary with the corresponding timelapse values
        summary_df = value_summary_df.merge(max_timelapse, on=group_list, how="left")
        
        summary_df.reset_index(inplace=True)
        
        file_name = f"global_dataset_max_summary_{measure}.csv"
        output_file = os.path.join(output_path, file_name)
        summary_df.to_csv(output_file, index=False)
        print(f"Saved global dataset max summary data to {output_file}")



# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


def gen_rep_summaries(df, chart_output_path, csv_output_path):
    """
    generate representation form summary data across all classifiers and all datasets for ML classifer test runs
    """
    print(f'\n\tgenerating representation form summary data for ML classifiers...')

    # Create output directories if they dont exist
    if not os.path.exists(chart_output_path):
        os.makedirs(chart_output_path)

    if not os.path.exists(csv_output_path):
        os.makedirs(csv_output_path)

    # Extract language model type and representation form from 'embeddings' column
    #df[['lm_type', 'representation_form']] = df['embeddings'].str.split(':', n=1, expand=True)

    df['lm_type'] = df.apply(lambda row: row['embeddings'].split(':')[0] if row['classifier'] in ML_CLASSIFIERS else row['embeddings'], axis=1)
    df['representation_form'] = df.apply(lambda row: row['embeddings'].split(':')[1] if row['classifier'] in ML_CLASSIFIERS else 'solo', axis=1)

    # Filter only the relevant measures
    df_filtered = df[df["measure"].isin(MEASURES)]

    # Selecting relevant columns for analysis
    df_filtered = df_filtered[["class_type", "classifier", "dataset", "lm_type", "representation_form", "measure", "value", "timelapse"]]

    # Pivoting the data to analyze performance metrics for each classifier and dataset
    df_pivot = df_filtered.pivot_table(index=["class_type", "classifier", "dataset", "lm_type", "representation_form"], 
                                    columns="measure", 
                                    values=["value", "timelapse"], 
                                    aggfunc="mean")

    # Reset index for better visualization
    df_pivot.reset_index(inplace=True)

    #
    # generate summary CSV fles
    #

    # Ensure all performance metric columns are numeric (excluding categorical columns)
    numeric_columns = df_pivot.select_dtypes(include=['number']).columns

    # Compute summary statistics by language model type, representation form, classifier, and dataset
    df_representation_summary = df_filtered.groupby(["classifier", "dataset", "representation_form"]).agg(
        mean_macro_f1=("value", "mean"),
        median_macro_f1=("value", "median"),
        mean_micro_f1=("value", "mean"),
        median_micro_f1=("value", "median"),
        mean_timelapse=("timelapse", "mean"),
        median_timelapse=("timelapse", "median")
    ).reset_index()

    # Save the results to CSV files
    output_path_pivot = os.path.join(csv_output_path, 'ml_classifier_summary.csv')
    output_path_rep_summary = os.path.join(csv_output_path, 'ml_representation_summary.csv')
    
    df_pivot.to_csv(output_path_pivot, index=False)
    df_representation_summary.to_csv(output_path_rep_summary, index=False)

    print(f"Analysis saved to {output_path_pivot}")
    print(f"Representation form summary saved to {output_path_rep_summary}")

    #
    # generate box plots
    #

    # Define legend patches
    legend_patches = [
        mpatches.Patch(color='lightblue', alpha=0.6, label='Timelapse (seconds)'),
        mpatches.Patch(color='C0', label='Measure Values')
    ]

    # Vertical box plots
    for measure, measure_label in zip(["mean_macro_f1", "mean_micro_f1"], ["Macro-F1 Score", "Micro-F1 Score"]):

        plt.figure(figsize=(10, 8))
        ax = sns.boxplot(y="representation_form", x=measure, data=df_representation_summary)
        ax2 = ax.twiny()
        sns.boxplot(y="representation_form", x="mean_timelapse", data=df_representation_summary, ax=ax2, color='lightblue', width=0.5, boxprops={'alpha': 0.6})
        ax2.set_xlabel("Timelapse (seconds)", fontsize=10)
        ax2.set_xlim(0, df_representation_summary["mean_timelapse"].max())
        ax.set_xlabel(measure_label, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        plt.yticks(fontsize=8)
        
        plt.title(f"{measure_label} and Timelapse by Representation Form (All Datasets and Classifiers)", fontsize=12)
        plt.legend(handles=legend_patches, loc='upper right')
        
        output_path_vertical_rep_summary = os.path.join(chart_output_path, f"representation_form_timelapse_boxplot_vertical_{measure}.png")
        plt.savefig(output_path_vertical_rep_summary)

        plt.close()

    # Horizontal box plots
    for measure, measure_label in zip(["mean_macro_f1", "mean_micro_f1"], ["Macro-F1 Score", "Micro-F1 Score"]):
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x="representation_form", y=measure, data=df_representation_summary)
        ax2 = ax.twinx()
        sns.boxplot(x="representation_form", y="mean_timelapse", data=df_representation_summary, ax=ax2, color='lightblue', width=0.5, boxprops={'alpha': 0.6})
        ax2.set_ylabel("Timelapse (seconds)", fontsize=10)
        ax2.set_ylim(0, df_representation_summary["mean_timelapse"].max())
        ax.set_ylabel(measure_label, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.title(f"{measure_label} and Timelapse by Representation Form (All Datasets and Classifiers)", fontsize=12)
        plt.legend(handles=legend_patches, loc='upper right')
        
        output_path_vertical_rep_summary = os.path.join(chart_output_path, f"representation_form_timelapse_boxplot_horizontal_{measure}.png")
        plt.savefig(output_path_vertical_rep_summary)

        plt.close()

    print(f"Box plots saved to {chart_output_path}")




# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
#
# final summary table data
#

def ml_summary_by_representation_form(df, output_path='../out', debug=False):
    """
    Summarizes measure values by representation form across language model families
    for ML classifiers, splitting embeddings into language model family and representation.
    Additionally provides a summary table broken out by dataset and representation form,
    including class_type and averaged timelapse values.

    Args:
        df (pd.DataFrame): Input DataFrame with classification results.
        output_path (str): Directory to save the summary TSVs.
        debug (bool): Flag to output debugging info.

    Returns:
        None: Writes summary TSVs to specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter ML classifiers
    df_ml = df[df['classifier'].isin(ML_CLASSIFIERS)].copy()

    if df_ml.empty:
        print("No data for ML classifiers.")
        return

    # Split embeddings into language model and representation form
    df_ml[['language_model_family', 'representation_form']] = df_ml['embeddings'].str.split(':', expand=True)

    # Group by representation form and measure, summarizing across language models
    summary_df = df_ml.groupby(['representation_form', 'measure']).agg(
        mean_value=('value', 'mean'),
        median_value=('value', 'median'),
        max_value=('value', 'max'),
        min_value=('value', 'min'),
        std_dev=('value', 'std'),
        count=('value', 'count'),
        mean_timelapse=('timelapse', 'mean')
    ).reset_index()

    # Sort for readability
    summary_df.sort_values(by=['representation_form', 'measure'], inplace=True)

    if debug:
        print(summary_df)

    # Save representation summary to TSV
    output_file_representation = os.path.join(output_path, 'summary_by_representation_form.tsv')
    summary_df.to_csv(output_file_representation, sep='\t', index=False)
    print(f"Summary by representation form saved to {output_file_representation}")

    # Additional breakout by dataset and representation form, including class_type and averaged timelapse
    dataset_summary_df = df_ml.groupby(['dataset', 'representation_form', 'class_type', 'measure']).agg(
        mean_value=('value', 'mean'),
        median_value=('value', 'median'),
        max_value=('value', 'max'),
        min_value=('value', 'min'),
        std_dev=('value', 'std'),
        count=('value', 'count'),
        mean_timelapse=('timelapse', 'mean')
    ).reset_index()

    dataset_summary_df.sort_values(by=['dataset', 'representation_form', 'measure'], inplace=True)

    if debug:
        print(dataset_summary_df)

    # Save dataset summary to TSV
    output_file_dataset = os.path.join(output_path, 'summary_by_dataset_representation_form.tsv')
    dataset_summary_df.to_csv(output_file_dataset, sep='\t', index=False)
    print(f"Summary by dataset and representation form saved to {output_file_dataset}")




def ml_classifier_topvalue_summary(df, output_path='../out', debug=False):
    """
    Outputs to a single TSV file the top 5 'value' rows for each measure,
    for each dataset, across all classifiers and representations.

    Args:
    - df: Pandas DataFrame containing the data.
    - output_path: Output directory for the resulting TSV file.
    - debug: Whether to print debug information.

    Returns:
    - None: Saves a single TSV file in the specified output directory.
    """

    # Filter for ML classifiers only
    ml_df = df[df['classifier'].isin(ML_CLASSIFIERS)].copy()

    if ml_df.empty:
        print("No data available for ML classifiers")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result_df = pd.DataFrame()

    # Group by dataset and measure
    grouped = ml_df.groupby(['dataset', 'measure'])

    for (dataset, measure), group_df in grouped:
        top_rows = group_df.nlargest(5, 'value')
        result_df = pd.concat([result_df, top_rows])

        if debug:
            print(f"Dataset: {dataset}, Measure: {measure}")
            print(top_rows)

    # Save to single TSV file
    output_file = os.path.join(output_path, "summary_topvalues_all.tsv")
    result_df.to_csv(output_file, sep='\t', index=False)

    print(f"\nTop 5 value summary (per dataset × measure) saved to {output_file}")


def summarize_dl_classifiers_by_classifier(df, output_path='../out', debug=False):
    """
    Summarizes measures for classifiers in DL_CLASSIFIERS by classifier.

    Args:
        df (pd.DataFrame): Input DataFrame with classification results.
        output_path (str): Directory to save the summary TSV.
        debug (bool): Flag to output debugging info.

    Returns:
        None: Writes summary TSV to specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter DL classifiers
    df_dl = df[df['classifier'].isin(DL_CLASSIFIERS)].copy()

    if df_dl.empty:
        print("No data for DL classifiers.")
        return

    # Group by classifier and measure, summarizing values
    summary_df = df_dl.groupby(['classifier', 'measure']).agg(
        mean_value=('value', 'mean'),
        median_value=('value', 'median'),
        max_value=('value', 'max'),
        min_value=('value', 'min'),
        std_dev=('value', 'std'),
        count=('value', 'count'),
        mean_timelapse=('timelapse', 'mean')
    ).reset_index()

    summary_df.sort_values(by=['classifier', 'measure'], inplace=True)

    if debug:
        print(summary_df)

    output_file = os.path.join(output_path, 'dl_classifier_summary_by_classifier.tsv')
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"DL classifier summary by classifier saved to {output_file}")


def summarize_all_classifiers_by_dataset(df, output_path='../out', debug=False):
    """
    Summarizes measures for classifiers in DL_CLASSIFIERS by dataset and classifier.

    Args:
        df (pd.DataFrame): Input DataFrame with classification results.
        output_path (str): Directory to save the summary TSV.
        debug (bool): Flag to output debugging info.

    Returns:
        None: Writes summary TSV to specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Group by dataset, classifier, and measure, summarizing values
    summary_df = df.groupby(['dataset', 'classifier', 'measure']).agg(
        mean_value=('value', 'mean'),
        median_value=('value', 'median'),
        max_value=('value', 'max'),
        min_value=('value', 'min'),
        std_dev=('value', 'std'),
        count=('value', 'count'),
        mean_timelapse=('timelapse', 'mean')
    ).reset_index()

    summary_df.sort_values(by=['dataset', 'classifier', 'measure'], inplace=True)

    if debug:
        print(summary_df)

    output_file = os.path.join(output_path, 'all_classifier_summary_by_dataset.tsv')
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"DL classifier summary by dataset saved to {output_file}")


def summarize_dl_classifiers_by_dataset(df, output_path='../out', debug=False):
    """
    Summarizes measures for classifiers in DL_CLASSIFIERS by dataset and classifier.

    Args:
        df (pd.DataFrame): Input DataFrame with classification results.
        output_path (str): Directory to save the summary TSV.
        debug (bool): Flag to output debugging info.

    Returns:
        None: Writes summary TSV to specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter DL classifiers
    df_dl = df[df['classifier'].isin(DL_CLASSIFIERS)].copy()

    if df_dl.empty:
        print("No data for DL classifiers.")
        return

    # Group by dataset, classifier, and measure, summarizing values
    summary_df = df_dl.groupby(['dataset', 'classifier', 'measure']).agg(
        mean_value=('value', 'mean'),
        median_value=('value', 'median'),
        max_value=('value', 'max'),
        min_value=('value', 'min'),
        std_dev=('value', 'std'),
        count=('value', 'count'),
        mean_timelapse=('timelapse', 'mean')
    ).reset_index()

    summary_df.sort_values(by=['dataset', 'classifier', 'measure'], inplace=True)

    if debug:
        print(summary_df)

    output_file = os.path.join(output_path, 'dl_classifier_summary_by_dataset.tsv')
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"DL classifier summary by dataset saved to {output_file}")


def summarize_all_by_classifier(df, output_path='../out', debug=False):
    """
    Summarizes measures for all data by classifier, includes ML and DL models.

    Args:
        df (pd.DataFrame): Input DataFrame with classification results.
        output_path (str): Directory to save the summary TSV.
        debug (bool): Flag to output debugging info.

    Returns:
        None: Writes summary TSV to specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Group by classifier and measure, summarizing values
    summary_df = df.groupby(['classifier', 'measure']).agg(
        mean_value=('value', 'mean'),
        median_value=('value', 'median'),
        max_value=('value', 'max'),
        min_value=('value', 'min'),
        std_dev=('value', 'std'),
        count=('value', 'count'),
        mean_timelapse=('timelapse', 'mean')
    ).reset_index()

    summary_df.sort_values(by=['classifier', 'measure'], inplace=True)

    if debug:
        print(summary_df)

    output_file = os.path.join(output_path, 'summary_by_classifier.tsv')
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"Summary by classifier saved to {output_file}")





def timelapse_correlation_scatter_plot_by_classifier(df, output_path='../out', debug=False):
    """
    Plots individual scatter charts showing correlation between training times (timelapse) and model performance
    for each measure defined in MEASURES, separately by dataset and measure, including all classifiers with different shapes.

    Args:
        df (pd.DataFrame): DataFrame containing classifier results.
        output_path (str): Directory to save the scatter plots.
        debug (bool): Debug flag for additional console output.

    Returns:
        None: Saves the plots to the specified directory.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define classifier shapes
    classifier_shapes = {
        'hf.sc': 'o', 'hf.cnn': 'o',       # Transformers
        'lr': 's', 'nb': 's', 'svm': 's',  # Classical ML
        'cnn': '^', 'attn': '^', 'lstm': '^'  # Deep Learning
    }

    # Filter the dataframe by the measures
    df_filtered = df[df['measure'].isin(MEASURES)].copy()

    if df_filtered.empty:
        print("No data for selected measures.")
        return

    for dataset in df_filtered['dataset'].unique():
        df_dataset = df_filtered[df_filtered['dataset'] == dataset]

        for measure in MEASURES:
            measure_df = df_dataset[df_dataset['measure'] == measure]

            if measure_df.empty:
                if debug:
                    print(f"No data available for dataset: {dataset}, measure: {measure}")
                continue

            plt.figure(figsize=(8, 6))  # Reduced figure size for clarity

            for classifier in measure_df['classifier'].unique():
                classifier_df = measure_df[measure_df['classifier'] == classifier]
                shape = classifier_shapes.get(classifier, 'X')  # Default shape if not found

                sns.scatterplot(
                    data=classifier_df,
                    x='timelapse',
                    y='value',
                    label=classifier,
                    marker=shape,
                    s=15,
                    alpha=0.8
                )

            plt.title(f'Correlation of Training Time to {measure} for {dataset}', fontsize=14)
            plt.xlabel('Training Time (seconds)', fontsize=12)
            plt.ylabel(f'{measure}', fontsize=12)
            plt.legend(title='Classifier', fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)

            plot_file = os.path.join(output_path, f'classifier_timelapse_performance_correlation_{dataset}_{measure}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')

            if debug:
                print(f"Scatter plot for {dataset}, measure {measure} saved to {plot_file}")



def timelapse_correlation_scatter_plot_by_model(df, output_path='../out', debug=False):
    """
    Plots individual scatter charts showing correlation between training times (timelapse) and model performance
    for each measure defined in MEASURES, separately by dataset and measure, using language model identity from the 'model' column.

    Args:
        df (pd.DataFrame): DataFrame containing classifier results.
        output_path (str): Directory to save the scatter plots.
        debug (bool): Debug flag for additional console output.

    Returns:
        None: Saves the plots to the specified directory.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Filter the dataframe by the measures
    df_filtered = df[df['measure'].isin(MEASURES)].copy()

    if df_filtered.empty:
        print("No data for selected measures.")
        return

    for dataset in df_filtered['dataset'].unique():
        df_dataset = df_filtered[df_filtered['dataset'] == dataset]

        for measure in MEASURES:
            measure_df = df_dataset[df_dataset['measure'] == measure]

            if measure_df.empty:
                if debug:
                    print(f"No data available for dataset: {dataset}, measure: {measure}")
                continue

            plt.figure(figsize=(8, 6))  # Reduced figure size for clarity

            for model in measure_df['model'].unique():
                model_df = measure_df[measure_df['model'] == model]

                sns.scatterplot(
                    data=model_df,
                    x='timelapse',
                    y='value',
                    label=model,
                    s=15,
                    alpha=0.8
                )

            plt.title(f'Correlation of Training Time to {measure} for {dataset}', fontsize=14)
            plt.xlabel('Training Time (seconds)', fontsize=12)
            plt.ylabel(f'{measure}', fontsize=12)
            plt.legend(title='Language Model', fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)

            plot_file = os.path.join(output_path, f'model_timelapse_performance_correlation_{dataset}_{measure}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')

            if debug:
                print(f"Scatter plot for {dataset}, measure {measure} saved to {plot_file}")



import plotly.express as px


def plotly_timelapse_correlation_scatter_plot_by_model(df, output_path='../out', debug=False):
    """
    Generates interactive HTML scatter plots showing correlation between training time (timelapse) and model performance,
    grouped by dataset and measure, colored by language model ('model' column), with mouseover tooltips.

    Args:
        df (pd.DataFrame): DataFrame containing classifier results.
        output_path (str): Directory to save the scatter plots.
        debug (bool): Debug flag for additional console output.

    Returns:
        None: Saves interactive plots as HTML files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_filtered = df[df['measure'].isin(MEASURES)].copy()

    if df_filtered.empty:
        print("No data for selected measures.")
        return

    for dataset in df_filtered['dataset'].unique():
        df_dataset = df_filtered[df_filtered['dataset'] == dataset]

        for measure in MEASURES:
            measure_df = df_dataset[df_dataset['measure'] == measure]

            if measure_df.empty:
                if debug:
                    print(f"No data for dataset: {dataset}, measure: {measure}")
                continue

            fig = px.scatter(
                measure_df,
                x='timelapse',
                y='value',
                color='model',
                hover_data=['model', 'classifier', 'representation', 'dimensions'],
                title=f'Training Time vs {measure} for {dataset}',
                labels={'value': measure, 'timelapse': 'Training Time (seconds)'}
            )

            fig.update_layout(
                height=600,
                width=900,
                title_font=dict(size=16),
                xaxis_title='Training Time (seconds)',
                yaxis_title=measure,
                legend_title_text='Language Model',
                template='plotly_white'
            )

            plot_file = os.path.join(output_path, f'model_timelapse_performance_correlation_{dataset}_{measure}.html')
            fig.write_html(plot_file)

            if debug:
                print(f"Interactive HTML plot saved to {plot_file}")



# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n\t----- Layer Cake Analysis & Reporting -----")
    
    parser = argparse.ArgumentParser(description="Layer Cake Analysis and Reporting Engine")

    parser.add_argument('file_path', type=str, help='Path to the TSV file with the data')
    parser.add_argument('-c', '--charts', action='store_true', default=False, help='Generate charts')
    parser.add_argument('-m', '--heatmaps', action='store_true', default=False, help='Generate heatmaps')
    parser.add_argument('-s', '--summary', action='store_true', default=False, help='Generate summary')
    parser.add_argument('-o', '--output_dir', action='store_true', help='Directory to write output files. If not provided, defaults to ' + OUT_DIR + ' + the base file name of the input file.')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('-y', '--ystart', type=float, default=Y_AXIS_THRESHOLD, help='Y-axis starting value for the charts (default: 0.6)')
    parser.add_argument('-r', '--results', type=int, default=TOP_N_RESULTS, help=f'number of results to display (default: {TOP_N_RESULTS})')
    parser.add_argument('-show', action='store_true', default=False, help='Display charts interactively (requires -c)')
    parser.add_argument('-a', '--analysis', action='store_true', default=False, help='Generate analysis data')

    args = parser.parse_args()
    print("args: ", args)

    # Ensure at least one operation is specified
    if not (args.charts or args.summary or args.analyze):
        parser.error("No action requested, add -c for charts or -s for summary or -a for analyze")

    debug = args.debug
    print("debug mode:", debug)

    print("y_start:", args.ystart)

    print("number of results:", args.results)

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

    summ_dir = os.path.join(out_dir, 'summary_data')
    analysis_dir = os.path.join(out_dir, 'analysis_data')
    
    charts_dir = os.path.join(out_dir, 'model_charts')
    heatmap_dir = os.path.join(out_dir, 'model_heatmaps')
    
    #
    # Check if output directories exist, otehrwise create them
    #
    if not os.path.exists(out_dir):

        print("\n")

        create_dir = input(f"Output directory {out_dir} does not exist. Do you want to create it? (y/n): ").strip().lower()
        
        if create_dir == 'y':
            os.makedirs(out_dir)
            os.makedirs(summ_dir)
            os.makedirs(analysis_dir)            
            os.makedirs(charts_dir)
            os.makedirs(heatmap_dir)
            
            print(f"Directory {out_dir} created, along with subdirectories")
        else:
            print(f"Directory {out_dir} was not created. Exiting.")
            exit()

    #
    # generate summary data across all classifiers and datasets
    #
    if args.summary:

        summarize_dl_classifiers_by_classifier(
            df,
            output_path=summ_dir,
            debug=args.debug
        )

        summarize_dl_classifiers_by_dataset(
            df,
            output_path=summ_dir,
            debug=args.debug
        )

        summarize_all_classifiers_by_dataset(
            df,
            output_path=summ_dir,
            debug=args.debug
        )

        summarize_all_by_classifier(
            df,
            output_path=summ_dir,
            debug=args.debug
        )

        ml_summary_by_representation_form(
            df,
            output_path=summ_dir,
            debug=args.debug
        )
         
        ml_classifier_topvalue_summary(
            df,
            output_path=summ_dir,
            debug=args.debug
        )

        perforamance_analysis_summary(
            df, 
            out_dir=summ_dir, 
            debug=args.debug
        )

        gen_rep_summaries(
            df=df,
            chart_output_path=summ_dir,
            csv_output_path=summ_dir
        )

        # gen_csvs(df, out_dir, neural=args.neural, debug=debug)
        gen_csvs_all(
            df,
            chart_output_dir=charts_dir, 
            csv_output_dir=analysis_dir, 
            debug=debug
        )
        
        gen_global_classifier_summaries(
            df=df, 
            output_path=summ_dir
        )

        gen_global_lm_summaries(
            df=df, 
            output_path=summ_dir
        )

        gen_global_dataset_max_data(
            df=df, 
            output_path=summ_dir
        )

        gen_global_dataset_summaries(
            df=df, 
            output_path=summ_dir
        )

    #
    # CSV and OUT data files for Excel manipulation and analysis
    #
    if (args.analysis):

        gen_summary_all(
            df=df, 
            output_path=analysis_dir, 
            gen_file=True,
            debug=debug
        )

        gen_dataset_summaries(
            df=df, 
            output_path=analysis_dir, 
            gen_file=True, 
            stdout=False, 
            debug=debug
        )

        timelapse_correlation_scatter_plot_by_classifier(
            df, 
            output_path=analysis_dir,
            debug=args.debug
        )

        timelapse_correlation_scatter_plot_by_model(
            df,
            output_path=analysis_dir,
            debug=args.debug
        )

        plotly_timelapse_correlation_scatter_plot_by_model(
            df,
            output_path=analysis_dir,
            debug=args.debug
        )
        
    #
    # generate charts
    #
    if args.charts:
            
        performance_analysis_detail(
            df, 
            out_dir=charts_dir, 
            debug=args.debug
        )

        model_performance_time_horizontal(
            df=df, 
            output_path=charts_dir,
            y_axis_threshold=args.ystart,
            top_n_results=args.results,
            show_charts=args.show,
            debug=debug
        )
        
        model_performance_time_vertical(
            df=df, 
            output_path=charts_dir,
            x_axis_threshold=args.ystart,
            top_n_results=args.results,
            show_charts=args.show,
            debug=debug
        )

        plotly_model_performance_dual_yaxis(
                df=df, 
                output_path=charts_dir, 
                y_axis_threshold=args.ystart,
                num_results=args.results,
                show_charts=args.show, 
                debug=debug
            )
         
        plotly_model_performance_horizontal(
                df=df, 
                output_path=charts_dir, 
                y_axis_threshold=args.ystart,
                num_results=args.results,
                show_charts=args.show, 
                debug=debug
            )
        
        all_model_performance_time_horizontal(
            df=df, 
            output_path=charts_dir,
            y_axis_threshold=args.ystart,
            top_n_results=args.results,
            show_charts=args.show,
            debug=debug
        )
        
        all_model_performance_time_vertical(
            df=df, 
            output_path=charts_dir,
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
            output_path=heatmap_dir,
            top_n_results=args.results,
            debug=debug
        )

        generate_vertical_heatmap_all_models(
            df, 
            output_path=heatmap_dir,
            top_n_results=args.results,
            debug=debug
        )

        generate_horizontal_heatmap_by_model(
            df, 
            output_path=heatmap_dir,
            top_n_results=args.results,
            debug=debug
        )

        generate_horizontal_heatmap_all_models(
            df, 
            output_path=heatmap_dir,
            top_n_results=args.results,
            debug=debug
        )
