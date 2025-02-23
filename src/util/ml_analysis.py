import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

LOG_PATH = '../log/test/'
INPUT_FILE = 'ml_def_full_test.test'
OUT_PATH = '../out/ml_analysis/'

# Define file path
file_path = LOG_PATH + INPUT_FILE

# Read the tab-delimited log file
df = pd.read_csv(file_path, sep="\t")

# Extract language model type and representation form from 'embeddings' column
df[['lm_type', 'representation_form']] = df['embeddings'].str.split(':', n=1, expand=True)

# Filter only the relevant measures
supported_measures = ["final-te-macro-f1", "final-te-micro-f1"]
df_filtered = df[df["measure"].isin(supported_measures)]

# Selecting relevant columns for analysis
df_filtered = df_filtered[["class_type", "classifier", "dataset", "lm_type", "representation_form", "measure", "value", "timelapse"]]

# Pivoting the data to analyze performance metrics for each classifier and dataset
df_pivot = df_filtered.pivot_table(index=["class_type", "classifier", "dataset", "lm_type", "representation_form"], 
                                   columns="measure", 
                                   values=["value", "timelapse"], 
                                   aggfunc="mean")

# Reset index for better visualization
df_pivot.reset_index(inplace=True)

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
output_path_pivot = OUT_PATH + 'ml_classifier_summary.csv'              # Update the path where you want to save the file
output_path_rep_summary = OUT_PATH + "ml_representation_summary.csv"

df_pivot.to_csv(output_path_pivot, index=False)
df_representation_summary.to_csv(output_path_rep_summary, index=False)

print(f"Analysis saved to {output_path_pivot}")
print(f"Representation form summary saved to {output_path_rep_summary}")



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
    plt.savefig(OUT_PATH + f"{measure}_timelapse_boxplot_vertical.png")
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
    plt.savefig(OUT_PATH + f"{measure}_timelapse_boxplot_horizontal.png")
    plt.close()

print(f"Box plots saved to {OUT_PATH}")
