# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:26:05 2024

Pipline to take the md and fa maps and the label files and find the md and fa values for regions of interest.

@author: nicol
"""


import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import logging
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import regtools


def load_nifti_data(file_path, return_img=False):
    """
    Load a NIfTI file from the specified path.
    """
    try:
        if return_img:
            data, affine, nifti = load_nifti(file_path, return_img=True)
            return data, affine, nifti
        else:
            data, affine = load_nifti(file_path)
            return data, affine
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")


def configure_logging(output_dir):
    """
    Configure logging to save logs in the given output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "process_log.txt")

    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def create_overlay(mni_template, transformed_image, output_path, title_prefix="Overlay"):
    """
    Create and save overlays of the MNI template and transformed image for axial, coronal, and sagittal views.
    """
    try:
        # Create overlays for three orthogonal slices
        slice_indices = {
            1: mni_template.shape[1] // 2,  # Coronal (choose middle slice along axis 1)
        }

        plt.figure(figsize=(15, 5))
        for axis, slice_index in slice_indices.items():
            plt.subplot(1, 3, axis + 1)
            regtools.overlay_slices(
                mni_template,
                transformed_image,
                None,
                axis,  # Axis index (0: Axial, 1: Coronal, 2: Sagittal)
                f"{title_prefix} (Slice {slice_index})",
                f"{title_prefix} (Transformed Slice {slice_index})"
            )

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
    except Exception as e:
        raise RuntimeError(f"Error creating overlay using regtools: {e}")



def extract_regions_of_interest(label_image, fa_map, md_map):
    """
    Extract FA and MD values for defined regions of interest.
    """
    atlas_labels = {
        8: "Corpus Callosum Right",
        68: "Corpus Callosum Left",
        12: "Internal Capsule Left",
        112: "Internal Capsule Right",
        6: "Hippocampus Right",
        106: "Hippocampus Left",
        4: "Thalamus Left",
        204: "Thalamus Right",
        209: "CC: EH Right",
        230: "CC: EH Left",
        64: "CC: FL Right",
        190: "CC: FL Left",
        130: "CC: OL Right",
        164: "CC: OL Left",
        181: "CC: PTL Right",
        180: "CC: PTL Left",
    }

    fa_md_stats = {}

    for label, region_name in atlas_labels.items():
        roi_mask = (label_image == label)
        fa_roi_values = fa_map[roi_mask]
        md_roi_values = md_map[roi_mask]

        if fa_roi_values.size > 0 and md_roi_values.size > 0:
            fa_mean = np.mean(fa_roi_values)
            fa_std = np.std(fa_roi_values)
            md_mean = np.mean(md_roi_values)
            md_std = np.std(md_roi_values)

            fa_md_stats[region_name] = {
                "FA Mean": fa_mean,
                "FA Std Dev": fa_std,
                "MD Mean": md_mean,
                "MD Std Dev": md_std,
            }
        else:
            fa_md_stats[region_name] = {
                "FA Mean": None,
                "FA Std Dev": None,
                "MD Mean": None,
                "MD Std Dev": None,
            }

    return fa_md_stats


def process_dataset(warp_md_path, warp_fa_path, label_path, output_dir):
    """
    Process a single dataset and save results.
    """
    logger = configure_logging(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Loading data...")
        md_data, md_affine = load_nifti_data(warp_md_path)
        fa_data, fa_affine = load_nifti_data(warp_fa_path)
        label_data, label_affine = load_nifti_data(label_path)


        # Check shapes of the images
        print("Label Image Shape:", label_data.shape)
        print("FA Map Shape:", fa_data.shape)
        print("MD Map Shape:", md_data.shape)

        # Ensure all images have the same orientation
        fa_data_t = np.transpose(fa_data, (0, 2, 1))  # Example adjustment
        md_data_t = np.transpose(md_data, (0, 2, 1))  # Example adjustment
        fa_map_flip = np.flip(fa_data_t, axis=2)  # Flip vertically (example)
        md_map_flip = np.flip(md_data_t, axis=2)  # Flip vertically (example)
        # Verify alignment
        print("Aligned FA Map Shape:", fa_map_flip.shape)
        print("Aligned MD Map Shape:", md_map_flip.shape)

        logger.info("Creating overlays...")
        mni_template = label_data  # Assuming label image is used as the template
        fa_overlay_path = os.path.join(output_dir, "fa_overlay.png")
        md_overlay_path = os.path.join(output_dir, "md_overlay.png")

        create_overlay(mni_template, fa_map_flip, fa_overlay_path, title_prefix="FA Overlay")
        create_overlay(mni_template, md_map_flip, md_overlay_path, title_prefix="MD Overlay")
        logger.info(f"Overlays saved to {output_dir}")

        logger.info("Extracting regions of interest...")
        fa_md_stats = extract_regions_of_interest(label_data, fa_map_flip, md_map_flip)

        stats_path = os.path.join(output_dir, "roi_stats.json")
        with open(stats_path, "w") as f:
            json.dump(fa_md_stats, f, indent=4)
        logger.info(f"ROI stats saved to {stats_path}")



    except Exception as e:
        logger.error(f"Error processing dataset: {e}")



def batch_process(root_dir, output_base_dir):
    """
    Batch process all datasets in the root directory.
    """
    dataset_dirs = glob.glob(os.path.join(root_dir, "*_loaded"))

    for dataset_dir in dataset_dirs:
        # Extract the third chunk of numbers from the dataset folder name
        dataset_name = os.path.basename(dataset_dir)
        try:
            third_chunk = dataset_name.split('_')[2]
        except IndexError:
            print(f"Skipping {dataset_dir}: Unable to parse the third chunk of numbers.")
            continue

        # Construct paths for the required files
        warp_md_path = os.path.join(dataset_dir, "patch2self_affine_False/warp_md.nii/warp_md.nii")
        warp_fa_path = os.path.join(dataset_dir, "patch2self_affine_False/warp_fa.nii/warp_fa.nii")
        label_path = os.path.join(
            root_dir,
            "Converted_diffusion_data",
            third_chunk,
            "X2P1_resize_bet4animal_warped_label.nii.gz"
        )

        if all(os.path.exists(p) for p in [warp_md_path, warp_fa_path, label_path]):
            output_dir = os.path.join(output_base_dir, dataset_name)
            process_dataset(warp_md_path, warp_fa_path, label_path, output_dir)
        else:
            print(f"Skipping {dataset_dir}: Missing one or more required files.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process FA and MD maps to extract ROI values.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing datasets.")
    parser.add_argument("--output_base_dir", required=True, help="Output base directory for processed results.")

    args = parser.parse_args()
    batch_process(args.root_dir, args.output_base_dir)

