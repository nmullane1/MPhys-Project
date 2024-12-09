# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:55:32 2024

@author: nicol
"""
import os
#import fsl.wrappers as fsw
import nibabel as nib
import numpy as np
os.environ['FSLDIR'] = R'\\wsl$\Ubuntu\home\mbcxamk2\fsl'
#DiPy
from dipy.denoise.localpca import mppca
import dipy.reconst.dti as dti
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.denoise.noise_estimate as ne
from dipy.align import motion_correction
from skimage.metrics import normalized_mutual_information as mi
from dipy.viz import regtools
from dipy.io.image import load_nifti, save_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align import affine_registration, register_dwi_to_template
from dipy.denoise.gibbs import gibbs_removal
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import color_fa
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.denoise.patch2self import patch2self
from dipy.denoise.noise_estimate import piesno
import matplotlib.pyplot as plt
from scipy.stats import norm
from dipy.align import resample
from dipy.viz.regtools import overlay_slices
import glob
import json
import logging

# Configure FSLDIR environment variable
os.environ['FSLDIR'] = R'\\wsl$\Ubuntu\home\mbcxamk2\fsl'



def configure_logging(output_dir):
    """
    Configure logging to save logs in the given output directory.
    Ensures that the directory exists before creating the log file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "process_log.txt")

    # Create a logger
    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.INFO)

    # Clear previous handlers (to avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler to write logs to a file in the output directory
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))  # No timestamps

    # Stream handler to output logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

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
        print(f"Error loading file {file_path}: {e}")
        return None, None

def load_method_file(method_path):
    """
    Load the method.npy file and extract B-values and B-vectors.
    """
    try:
        method = np.load(method_path, allow_pickle=True).item()
        bval = method['PVM_DwEffBval']
        bvec = method['PVM_DwGradVec']
        return bval, bvec
    except Exception as e:
        print(f"Error loading method file {method_path}: {e}")
        return None, None

def perform_motion_correction_with_retries(nifti, bval, bvec, affine, max_retries=3):
    """
    Perform motion correction on diffusion data with retries for failures.
    """
    try:
        gtab = gradient_table(bval, bvec, atol=1)
        pipeline_variants = [
            ["center_of_mass", "translation", "rigid"],
            ["translation", "rigid"],
            ["rigid"]
        ]
        for attempt, pipeline in enumerate(pipeline_variants[:max_retries], start=1):
            try:
                print(f"Attempt {attempt}: Trying motion correction pipeline {pipeline}...")
                dwi_corrected, reg_affines = motion_correction(nifti, gtab, affine, pipeline=pipeline)
                print("Motion correction completed successfully.")
                return dwi_corrected, reg_affines
            except Exception as e:
                print(f"Motion correction failed on attempt {attempt} with pipeline {pipeline}: {e}")
        raise RuntimeError("Motion correction failed after retries.")
    except Exception as e:
        print(f"Error during motion correction: {e}")
        return None, None


def perform_patch2self_denoising(data, bvals, nifti, save_path):
    """
    Perform denoising using the Patch2Self algorithm and save the denoised image.
    """
    try:
        print("Starting Patch2Self denoising...")
        image_denoised = patch2self(
            data, bvals, model='ols', shift_intensity=True,
            clip_negative_vals=False, b0_threshold=50
        )
        nib.save(nib.Nifti1Image(image_denoised, nifti.affine, header=nifti.header), save_path)
        print(f"Denoised image saved to: {save_path}")
        image_denoised = nib.load(save_path)
        image_denoised_data = image_denoised.get_fdata()
        print("The mean of the image_denoised data is:", np.mean(image_denoised_data))
        return image_denoised_data
    except Exception as e:
        print(f"Error during Patch2Self denoising: {e}")
        return None

def fit_tensor_model(bval, bvec):
    """
    Fit the diffusion tensor model using B-values and B-vectors.
    """
    try:
        print("Fitting Tensor Model...")
        gtab = gradient_table(bval, bvec, atol=1)
        tenmodel = dti.TensorModel(gtab)
        print(gtab)
        print("Tensor Model fitted successfully.")
        return tenmodel
    except Exception as e:
        print(f"Error fitting Tensor Model: {e}")
        return None

def perform_gibbs_correction(denoised_image_path, save_path, nifti):
    """
    Perform Gibbs correction on denoised data and save the corrected image.
    """
    try:
        print("Starting Gibbs correction...")
        image_denoised = nib.load(denoised_image_path)
        image_denoised_data = image_denoised.get_fdata()
        data_corrected = gibbs_removal(image_denoised_data, slice_axis=2, num_processes=-1)
        nib.save(nib.Nifti1Image(data_corrected, nifti.affine, header=nifti.header), save_path)
        print(f"Gibbs-corrected image saved to: {save_path}")
        return data_corrected
    except Exception as e:
        print(f"Error during Gibbs correction: {e}")
        return None


def calculate_mi_scores(diffusion_data, dwi_motion_corrected_data, image_denoised_data, data_corrected, output_dir, logger):
    """
    Calculate Mutual Information (MI) scores for each correction stage.
    """
    try:
        # Calculate the reference image as the mean of the first 5 b=0 images
        reference_image = np.mean(diffusion_data[..., :5], axis=-1)

        # Lists to store MI scores
        mi_scores_before_any_correction = []
        mi_scores_after_motion_correction = []
        mi_scores_after_denoise_correction = []
        mi_scores_after_gibbs_correction = []

        # Loop over the 24 b=1000 images (indices 5 to 28)
        for i in range(5, 29):  # 5 to 28 inclusive (b=1000 images)
            original_slice = diffusion_data[..., i]
            motion_correction_slice = dwi_motion_corrected_data[..., i]
            corrected_slice = image_denoised_data[..., i]
            gibbs_slice = data_corrected[..., i]

            # Calculate MI between the original uncorrected image and the reference image (before correction)
            mi_score_before = mi(reference_image, original_slice)
            mi_scores_before_any_correction.append(mi_score_before)

            # Calculate MI between the original and motion-corrected image (after correction)
            mi_score_after_motion = mi(reference_image, motion_correction_slice)
            mi_scores_after_motion_correction.append(mi_score_after_motion)

            # Calculate MI between the denoised corrected image and the reference image
            mi_score_after_denoise = mi(reference_image, corrected_slice)
            mi_scores_after_denoise_correction.append(mi_score_after_denoise)

            # Calculate MI between the Gibbs-corrected image and the reference image
            mi_score_after_gibbs = mi(reference_image, gibbs_slice)
            mi_scores_after_gibbs_correction.append(mi_score_after_gibbs)

        # Compute average MI scores across all 24 b=1000 images
        average_mi_before = np.mean(mi_scores_before_any_correction)
        average_mi_after_motion = np.mean(mi_scores_after_motion_correction)
        average_mi_after_denoise = np.mean(mi_scores_after_denoise_correction)
        average_mi_after_gibbs = np.mean(mi_scores_after_gibbs_correction)

        # Print results
        logger.info("printing MI")
        logger.info(f"Average MI score across the 24 b=1000 images before any corrections: {average_mi_before}")
        logger.info(f"Average MI score across the 24 b=1000 images after motion correction: {average_mi_after_motion}")
        logger.info(f"Average MI score across the 24 b=1000 images after denoising correction: {average_mi_after_denoise}")
        logger.info(f"Average MI score across the 24 b=1000 images after Gibbs correction: {average_mi_after_gibbs}")
    except Exception as e:
        print(f"Error during MI calculation: {e}")

def plot_noise_histogram(data_corrected, mask, output_fig):
    """
    Plot a histogram of noise values extracted from the data, fit with a Gaussian distribution.

    Args:
    data_corrected : np.ndarray
        Corrected DWI data.
    mask : np.ndarray
        Mask generated using PIESNO.
    """
    try:
        # Extract noise values using the mask and clean the data
        noise_values = data_corrected[mask > 0]
        noise_values = noise_values[np.isfinite(noise_values)]  # Remove NaN and Inf values
        noise_values = noise_values[noise_values >= 0]          # Remove negative values
        # Fit the Gaussian (Normal) distribution to the noise values
        mean_norm, std_norm = norm.fit(noise_values)
        # Plot histogram of noise values
        plt.figure(figsize=(10, 6))
        plt.hist(noise_values.ravel(), bins=50, density=True, alpha=0.6, color='blue', label='Noise Histogram')
        # Generate Gaussian PDF using fitted parameters
        x_vals = np.linspace(noise_values.min(), noise_values.max(), 100)
        pdf_norm = norm.pdf(x_vals, mean_norm, std_norm)
        plt.plot(x_vals, pdf_norm, color='green', linewidth=2, linestyle='--',
                 label=f'Gaussian Fit (μ={mean_norm:.2f}, σ={std_norm:.2f})')
        # Customize plot
        plt.xlabel("Noise Value")
        plt.ylabel("Density")
        plt.title("Histogram of Noise Values with Gaussian Fit")
        plt.legend()
        plt.savefig(output_fig)
    except Exception as e:
        print(f"Error during noise histogram plotting: {e}")
def perform_tensor_model_fit(data_corrected, diffusion_affine, nifti, gtab, save_path_fa, save_path_md, save_path_evals, save_path_evecs, output_dir, logger):
    """
    Perform the tensor model fit and save the results as NIfTI files.
    """
    try:
        # Fit the tensor model
        tenmodel = dti.TensorModel(gtab)
        fit = tenmodel.fit(data_corrected)

        # Get DTI results
        fa = fit.fa
        md = fit.md
        evals = fit.evals
        evecs = fit.evecs

        # Save all results to `data/dti` as NIfTI files
        nib.save(nib.Nifti1Image(fa, diffusion_affine, header=nifti.header), save_path_fa)
        nib.save(nib.Nifti1Image(md, diffusion_affine, header=nifti.header), save_path_md)
        nib.save(nib.Nifti1Image(evals, diffusion_affine, header=nifti.header), save_path_evals)
        nib.save(nib.Nifti1Image(evecs, diffusion_affine, header=nifti.header), save_path_evecs)


        # Load results to calculate mean values
        fa_nifti = nib.load(save_path_fa)
        md_nifti = nib.load(save_path_md)

        fa_array = fa_nifti.get_fdata()
        md_array = md_nifti.get_fdata()
        mean_fa=np.mean(fa_array)
        mean_md=np.mean(md_array)
        # Print mean FA and MD values
        logger.info(f"The FA mean is: {mean_fa}")
        logger.info(f"The MD mean is: {mean_md}")
    except Exception as e:
        print(f"Error during tensor model fitting: {e}")

def diffusion_and_fa_md_to_T2(diffusion_data, diffusion_affine, T2_data, T2_affine, fa, md, save_path_warp_fa, save_path_warp_md, nifti):
    """
    Perform transformation from diffusion space to T2 space using affine and non-linear registration.
    Args:
    diffusion_data : np.ndarray
        4D diffusion data.
    diffusion_affine : np.ndarray
        4x4 affine matrix for diffusion data.
    T2_data : np.ndarray
        3D T2-weighted image.
    T2_affine : np.ndarray
        4x4 affine matrix for T2 data.
    Returns:
    tuple:
        - upsampled_diffusion_data: Diffusion data after affine transformation.
        - warped_moving: Diffusion data after non-linear transformation.
    """
    try:
        # Step 1: Affine Registration to Get Transformation Matrix
        nbins = 32  # Number of bins to discretize intensity distributions
        sampling_prop = None  # Full sampling
        metric = MutualInformationMetric(nbins, sampling_prop)
        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
        pipeline = ["center_of_mass", "translation", "rigid"]

        for n in range(0, 1):
            moving = diffusion_data[..., n]
            xformed_img, reg_affine = affine_registration(
                moving,
                T2_data,
                moving_affine=diffusion_affine,
                static_affine=T2_affine,
                nbins=32,
                metric="MI",
                pipeline=pipeline,
                level_iters=level_iters,
                sigmas=sigmas,
                factors=factors,
            )
        print("Affine registration completed. Transformation matrix obtained.")

        # Step 2: Apply Affine Transformation
        diff_data = diffusion_data[..., 0]
        fa_data=fa.get_fdata()
        md_data=md.get_fdata()

        affine_map = AffineMap(
            reg_affine, T2_data.shape, T2_affine,
            diff_data.shape, diffusion_affine
        )
        upsampled_diffusion_data = affine_map.transform(diff_data)
        print("Affine transformation applied to diffusion data.")
        # Step 3: Symmetric Diffeomorphic Registration
        metric = CCMetric(3)
        level_iters = [10, 10, 5]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        mapping = sdr.optimize(T2_data, diff_data, T2_affine, diffusion_affine, reg_affine)
        warped_moving = mapping.transform(diff_data)
        warped_fa=affine_map.transform(fa_data)
        warped_md=affine_map.transform(md_data)
        nib.save(nib.Nifti1Image(warped_fa, diffusion_affine, header=nifti.header), save_path_warp_fa)
        nib.save(nib.Nifti1Image(warped_md, diffusion_affine, header=nifti.header), save_path_warp_md)
        print("Non-linear registration completed. Diffusion data warped to T2 space.")

        return upsampled_diffusion_data, warped_moving
    except Exception as e:
        print(f"Error during diffusion to T2 transformation: {e}")
        return None, None



def cross_correlation(img1, img2):
    """
    Computes the cross-correlation between two images (3D volumes).
    Args:
    img1 : np.ndarray
        First image (e.g., warped diffusion data).
    img2 : np.ndarray
        Second image (e.g., T2 data).
    Returns:
    float
        Cross-correlation value.
    """
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape to compute cross-correlation.")

    # Flatten the arrays for easier computation
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Compute the means of each image
    mean_img1 = np.mean(img1_flat)
    mean_img2 = np.mean(img2_flat)

    # Subtract the means
    img1_zero_mean = img1_flat - mean_img1
    img2_zero_mean = img2_flat - mean_img2

    # Compute the numerator and denominators for the cross-correlation formula
    numerator = np.sum(img1_zero_mean * img2_zero_mean)
    denominator = np.sqrt(np.sum(img1_zero_mean ** 2) * np.sum(img2_zero_mean ** 2))

    # Return the cross-correlation
    if denominator == 0:
        return 0  # Avoid division by zero
    else:
        return numerator / denominator


def process_dataset(diffusion_path, T2_path, output_dir):
    """
    Process a single DTI and T2 dataset to generate FA and MD maps.
    """
    logger = configure_logging(output_dir)
    try:
        print("And awayyyyy we go!")
        # File paths
        denoised_save_path = os.path.join(output_dir, "image_denoised_temp.nii.gz")
        gibbs_save_path = os.path.join(output_dir, "image_gibbs.nii.gz")
        mask_save_path = os.path.join(output_dir, "mask_piesno.nii.gz")
        fa_path = os.path.join(output_dir, "fa.nii.gz")
        md_path = os.path.join(output_dir, "md.nii.gz")
        evals_path = os.path.join(output_dir, "evals.nii.gz")
        evecs_path = os.path.join(output_dir, "evecs.nii.gz")
        output_fig = os.path.join(output_dir, "noise.png")
        fa_warp_path = os.path.join(output_dir, "warp_fa.nii.gz")
        md_warp_path = os.path.join(output_dir, "warp_md.nii.gz")
        os.makedirs(output_dir, exist_ok=True)

        # Load diffusion and T2 data
        try:
            diffusion_data, diffusion_affine, diffusion_nifti = load_nifti_data(diffusion_path, return_img=True)
            T2_data, T2_affine, T2_nifti = load_nifti_data(T2_path, return_img=True)
            print("loading")
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        dataset_dir = os.path.abspath(os.path.join(diffusion_path, "../../../../"))
        method_file_path = os.path.join(dataset_dir, "method.npy")

        # Check if the method file exists
        if not os.path.exists(method_file_path):
            print(f"Method file not found in {method_file_path}. Skipping dataset.")
            return

        try:
            bval, bvec = load_method_file(method_file_path)
            if bval is None or bvec is None:
                raise ValueError("B-values or B-vectors are missing.")
        except Exception as e:
            print(f"Error loading method file {method_file_path}: {e}")
            return

        # Gradient table
        gtab = gradient_table(bval, bvec, atol=1)
        bvals=np.array(bval)
        bvals = np.array(bval)

        if bval is not None and bvec is not None:
            print("B-values and B-vectors loaded successfully.")

              # Motion correction with retries
            dwi_motion_corrected, reg_affines = perform_motion_correction_with_retries(diffusion_nifti, bval, bvec, diffusion_affine)
            if dwi_motion_corrected is None:
                logger.error("Motion correction failed. Skipping this dataset.")
                return

            if dwi_motion_corrected is not None:
                dwi_motion_corrected_data = dwi_motion_corrected.get_fdata()

                # Perform Patch2Self denoising
                denoised_data = perform_patch2self_denoising(
                    dwi_motion_corrected_data, bvals, diffusion_nifti, denoised_save_path
                )

                if denoised_data is not None:
                    # Perform Gibbs correction
                    gibbs_corrected_data = perform_gibbs_correction(
                        denoised_save_path, gibbs_save_path, diffusion_nifti
                    )

                    # Noise estimation using PIESNO
                    sigma, mask = piesno(gibbs_corrected_data, N=4, return_mask=True)
                    save_nifti(mask_save_path, mask.astype(np.uint8), diffusion_affine)
                    mean_sigma=np.mean(sigma)
                    std_background=np.std(gibbs_corrected_data[mask[..., :].astype(bool)])
                    logger.info(f"The noise standard deviation is sigma = {sigma}")
                    logger.info(f"The std of the background is: {std_background}")
                    logger.info(f"The mean of the sigma array: {mean_sigma}")
                    # Plot noise histogram
                    plot_noise_histogram(gibbs_corrected_data, mask, output_fig)
                    # Calculate mean signal and SNR
                    axial = gibbs_corrected_data[:, :, gibbs_corrected_data.shape[2] // 2, 0].T
                    mean_sig = np.mean(axial)
                    snr = mean_sig / np.mean(sigma)
                    logger.info(f'The mean signal is ={mean_sig}')
                    logger.info(f'The SNR is = {snr}')
                    # Calculate MI scores
                    calculate_mi_scores(
                        diffusion_data,
                        dwi_motion_corrected_data,
                        denoised_data,
                        gibbs_corrected_data, output_dir, logger
                    )

                    # Perform Tensor Model Fit
                    perform_tensor_model_fit(
                        gibbs_corrected_data, diffusion_affine, diffusion_nifti, gtab, fa_path, md_path, evals_path, evecs_path,output_dir, logger
                    )

                    fa_nifti = nib.load(fa_path)
                    md_nifti = nib.load(md_path)

                    # Transform diffusion images to T2
                    upsampled_diffusion_data, warped_moving = diffusion_and_fa_md_to_T2(
                        diffusion_data, diffusion_affine, T2_data, T2_affine, fa_nifti, md_nifti, fa_warp_path, md_warp_path, diffusion_nifti
                    )

                    if upsampled_diffusion_data is not None and warped_moving is not None:
                        logger.info("Transformation and warping to T2 completed successfully.")

                        # Calculate cross-correlation before and after warping
                        cross_corr_value_before = cross_correlation(upsampled_diffusion_data, T2_data)
                        logger.info(f"Cross-Correlation Value (Before Warping): {cross_corr_value_before}")

                        cross_corr_value_after = cross_correlation(warped_moving, T2_data)
                        logger.info(f"Cross-Correlation Value (After Warping): {cross_corr_value_after}")

    except Exception as e:
        print(f"Error processing dataset {diffusion_path}: {e}")


def batch_process(root_dir, output_base_dir):
    """
    Batch process all datasets in the root directory.
    """
    # Find all DTI and T2 file pairs
    dataset_dirs = glob.glob(os.path.join(root_dir, "Missing data","*_loaded"))
    for dataset_dir in dataset_dirs:
        diffusion_path = os.path.join(dataset_dir, "6", "pdata", "1", "niiobj_1.nii", "niiobj_1.nii")
        T2_path = os.path.join(dataset_dir, "2", "pdata", "1", "niiobj_1.nii", "niiobj_1.nii")
        if os.path.exists(diffusion_path) and os.path.exists(T2_path):
            output_dir = os.path.join(output_base_dir, os.path.basename(dataset_dir))
            process_dataset(diffusion_path, T2_path, output_dir)
        else:
            print(f"Skipping dataset {dataset_dir}: Missing required files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process MRI datasets to generate FA and MD maps.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing DTI and T2 datasets")
    parser.add_argument("--output_base_dir", required=True, help="Base directory to save outputs")

    args = parser.parse_args()
    batch_process(args.root_dir, args.output_base_dir)