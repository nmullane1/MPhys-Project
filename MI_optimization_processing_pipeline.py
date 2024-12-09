# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:39:47 2024

Code to extract MI and SNR information from optimization

@author: nicol
"""



import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib.ticker as ticker

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping of original pipeline names to display names
PIPELINE_NAME_MAPPING = {
    "patch2self_affine_False": "Patch2Self (Affine=False)",
    "patch2self_affine_True": "Patch2Self (Affine=True)",
    "mppca_affine_False": "MPPCA (Affine=False)",
    "mppca_affine_True": "MPPCA (Affine=True)",
}

# Define display-friendly names for plotting
STEP_DISPLAY_NAMES = {
    "before any corrections": "Before Any Corrections",
    "after motion correction": "After Motion Correction",
    "after denoising correction": "After Denoising Correction",
    "after Gibbs correction": "After Gibbs Correction",
}

def extract_mi_values(log_file):
    """
    Extract mutual information (MI) values from a single log file.
    """
    try:
        steps = [
            "before any corrections",
            "after motion correction",
            "after denoising correction",
            "after Gibbs correction",
        ]
        pipelines = {}
        with open(log_file, "r") as file:
            lines = file.readlines()

        current_pipeline = None
        for line in lines:
            if "Running pipeline:" in line:
                current_pipeline = line.strip().split(":")[1].strip()
                if current_pipeline not in pipelines:
                    pipelines[current_pipeline] = {step: [] for step in steps}

            if "Average MI score" in line and current_pipeline:
                for step in steps:
                    if step in line:
                        match = re.search(r": ([\d.]+)$", line.strip())
                        if match:
                            mi_value = float(match.group(1))
                            pipelines[current_pipeline][step].append(mi_value)
                        else:
                            logger.warning(f"Failed to extract MI value from line: {line.strip()}")

        return pipelines
    except Exception as e:
        logger.error(f"Error extracting MI values: {e}")
        return {}


def extract_snr_values(log_file):
    """
    Extract SNR values from a single log file.
    """
    try:
        snr_values = {}
        with open(log_file, "r") as file:
            lines = file.readlines()

        current_pipeline = None
        for line in lines:
            if "Running pipeline:" in line:
                current_pipeline = line.strip().split(":")[1].strip()
                if current_pipeline not in snr_values:
                    snr_values[current_pipeline] = []

            if "SNR=" in line and current_pipeline:
                match = re.search(r"SNR=([\d.]+)", line.strip())
                if match:
                    snr = float(match.group(1))
                    snr_values[current_pipeline].append(snr)

        return snr_values
    except Exception as e:
        logger.error(f"Error extracting SNR values: {e}")
        return {}


def compute_aggregated_stats(all_pipelines, metric="MI"):
    """
    Compute aggregated mean and standard deviation for all pipelines and steps (for MI) or overall values (for SNR).
    """
    aggregated_stats = {}
    for pipeline, steps_data in all_pipelines.items():
        if metric == "MI":
            aggregated_stats[pipeline] = {}
            for step, values in steps_data.items():
                aggregated_values = [v for dataset_values in values for v in dataset_values]
                aggregated_stats[pipeline][step] = {
                    "mean": np.mean(aggregated_values),
                    "std": np.std(aggregated_values),
                }
        elif metric == "SNR":
            aggregated_stats[pipeline] = {
                "mean": np.mean(steps_data),
                "std": np.std(steps_data),
            }

    return aggregated_stats


def process_all_datasets(root_dir):
    """
    Process all datasets in the root directory to compute aggregated MI and SNR stats.
    """
    try:
        dataset_dirs = glob.glob(os.path.join(root_dir, "*_loaded"))

        all_mi_values = {}
        all_snr_values = {}

        for dataset_dir in dataset_dirs:
            log_file = os.path.join(dataset_dir, "process_log.txt")
            if os.path.exists(log_file):
                dataset_mi_values = extract_mi_values(log_file)
                for pipeline, steps_data in dataset_mi_values.items():
                    if pipeline not in all_mi_values:
                        all_mi_values[pipeline] = {step: [] for step in steps_data}
                    for step, values in steps_data.items():
                        all_mi_values[pipeline][step].append(values)

                dataset_snr_values = extract_snr_values(log_file)
                for pipeline, snr_list in dataset_snr_values.items():
                    if pipeline not in all_snr_values:
                        all_snr_values[pipeline] = []
                    all_snr_values[pipeline].extend(snr_list)
            else:
                logger.warning(f"Skipping {dataset_dir}: Missing log file.")

        aggregated_mi_stats = compute_aggregated_stats(all_mi_values, metric="MI")
        aggregated_snr_stats = compute_aggregated_stats(all_snr_values, metric="SNR")

        return aggregated_mi_stats, aggregated_snr_stats
    except Exception as e:
        logger.error(f"Error in processing all datasets: {e}")
        return {}, {}


def plot_stats(aggregated_stats, output_dir, metric):
    """
    Plot aggregated statistics for all pipelines and steps (MI or SNR).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        steps = [
            "before any corrections",
            "after motion correction",
            "after denoising correction",
            "after Gibbs correction",
        ] if metric == "MI" else None
        display_steps = [STEP_DISPLAY_NAMES[step] for step in steps] if steps else None

        plt.figure(figsize=(12, 8))
        plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14})

        for pipeline, stats in aggregated_stats.items():
            display_name = PIPELINE_NAME_MAPPING.get(pipeline, pipeline)
            if metric == "MI":
                means = [stats[step]["mean"] for step in steps]
                stds = [stats[step]["std"] for step in steps]
                plt.errorbar(display_steps, means, yerr=stds, label=display_name, capsize=5, fmt='o-', linewidth=2, markersize=8)
            elif metric == "SNR":
                plt.bar(display_name, stats["mean"], yerr=stats["std"], capsize=5)

        plt.title(f"Aggregated {metric} Across Pipelines", fontsize=18)
        plt.xlabel("Processing Steps" if metric == "MI" else "Pipelines", fontsize=16)
        plt.ylabel(f"{metric} (MI)" if metric == "MI" else f"{metric} (SNR)", fontsize=16)
        plt.xticks(rotation=45 if metric == "MI" else 0)

        plt.legend(title="Pipelines" if metric == "MI" else None, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
        plt.tight_layout()
        plot_name = f"aggregated_pipeline_{metric.lower()}_analysis"
        plt.savefig(os.path.join(output_dir, f"{plot_name}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{plot_name}.pdf"))
        plt.close()
    except Exception as e:
        logger.error(f"Error in plotting {metric} statistics: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and compute MI and SNR statistics across all datasets.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing datasets.")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots and results.")

    args = parser.parse_args()

    aggregated_mi_stats, aggregated_snr_stats = process_all_datasets(args.root_dir)

    print("\nPipeline-specific MI Statistics:")
    for pipeline, stats in aggregated_mi_stats.items():
        print(f"{pipeline}: {stats}")

    print("\nPipeline-specific SNR Statistics:")
    for pipeline, stats in aggregated_snr_stats.items():
        print(f"{pipeline}: Mean SNR = {stats['mean']:.2f}, Std Dev = {stats['std']:.2f}")

    plot_stats(aggregated_mi_stats, args.output_dir, metric="MI")
    plot_stats(aggregated_snr_stats, args.output_dir, metric="SNR")
