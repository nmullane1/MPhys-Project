#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:35:32 2024

@author: alexbralsford
"""

import os
#import fsl.wrappers as fsw
import nibabel as nib
import numpy as np
import napari
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
import matplotlib.pyplot as plt

# Configure FSLDIR environment variable
os.environ['FSLDIR'] = R'\\wsl$\Ubuntu\home\mbcxamk2\fsl'

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

def perform_motion_correction(nifti, bval, bvec, affine, use_affine=False):
    """
    Perform motion correction on diffusion data.

    Parameters:
    - nifti: NIfTI object containing the diffusion-weighted data.
    - bval: List or array of B-values.
    - bvec: List or array of B-vectors.
    - affine: The affine transformation matrix of the input image.
    - use_affine: bool, whether to include the "affine" step in motion correction.

    Returns:
    - dwi_corrected: NIfTI object of the motion-corrected diffusion-weighted data.
    - reg_affines: List of affine transformations applied during motion correction.
    """
    try:
        # Define the pipeline steps
        motion_correction_pipeline = ["center_of_mass", "translation", "rigid"]
        if use_affine:
            motion_correction_pipeline.append("affine")

        # Generate gradient table
        gtab = gradient_table(bval, bvec, atol=1)

        # Perform motion correction
        dwi_corrected, reg_affines = motion_correction(
            nifti,
            gtab,
            affine,
            pipeline=motion_correction_pipeline
        )

        print(f"Motion correction completed successfully with {'affine' if use_affine else 'non-affine'} pipeline.")
        return dwi_corrected, reg_affines

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

def perform_mppca_denoising(data, nifti, save_path):
    """
    Perform denoising using the MPPCA algorithm and save the denoised image.
    """
    try:
        print("Starting MPPCA denoising...")
        image_denoised_temp = mppca(data)
        nib.save(nib.Nifti1Image(image_denoised_temp, nifti.affine, header=nifti.header), save_path)
        print(f"Denoised image saved to: {save_path}")
        image_denoised_temp = nib.load(save_path)
        image_denoised_temp_data = image_denoised_temp.get_fdata()
        print("The mean of the image_denoised_temp_data is:", np.mean(image_denoised_temp_data))
        return image_denoised_temp
    except Exception as e:
        print(f"Error during MPPCA denoising: {e}")
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
    

def calculate_mi_scores(diffusion_data, dwi_motion_corrected_data, image_denoised_data, data_corrected):
    """
    Calculate Mutual Information (MI) scores for each correction stage.
    """
    try:
        # Calculate the reference image as the mean of the first 5 b=0 images
        reference_image = np.mean(diffusion_data[..., :5], axis=-1)

        # Convert reference_image to a NumPy array if needed
        if not isinstance(reference_image, np.ndarray):
            reference_image = reference_image.get_fdata()

        # Convert all inputs to NumPy arrays
        diffusion_data = diffusion_data.get_fdata() if hasattr(diffusion_data, 'get_fdata') else diffusion_data
        dwi_motion_corrected_data = dwi_motion_corrected_data.get_fdata() if hasattr(dwi_motion_corrected_data, 'get_fdata') else dwi_motion_corrected_data
        image_denoised_data = image_denoised_data.get_fdata() if hasattr(image_denoised_data, 'get_fdata') else image_denoised_data
        data_corrected = data_corrected.get_fdata() if hasattr(data_corrected, 'get_fdata') else data_corrected

        # Lists to store MI scores
        mi_scores_before_any_correction = []
        mi_scores_after_motion_correction = []
        mi_scores_after_denoise_correction = []
        mi_scores_after_gibbs_correction = []

        # Loop over the 24 b=1000 images (indices 5 to 28)
        for i in range(5, 29):  # 5 to 28 inclusive (b=1000 images)
            # Extract slices as NumPy arrays
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
        
        print("The Average MI before is:", average_mi_before)
        print("The Average MI after Motion Correction is:", average_mi_after_motion)
        print("The Average MI after Denoising is:", average_mi_after_denoise)
        print("The Average MI after Gibbs Ringing Removal is:", average_mi_after_gibbs)
        

        # Return results as a dictionary
        return {
            "average_mi_before": average_mi_before,
            "average_mi_after_motion": average_mi_after_motion,
            "average_mi_after_denoise": average_mi_after_denoise,
            "average_mi_after_gibbs": average_mi_after_gibbs
        }
    except Exception as e:
        print(f"Error during MI calculation: {e}")
        return None



def perform_tensor_model_fit(data_corrected, diffusion_affine, nifti, gtab):
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
        nib.save(nib.Nifti1Image(fa, diffusion_affine, header=nifti.header), R"data\dti\fa.nii.gz")
        nib.save(nib.Nifti1Image(md, diffusion_affine, header=nifti.header), R"data\dti\md.nii.gz")
        nib.save(nib.Nifti1Image(evals, diffusion_affine, header=nifti.header), R"data\dti\evals.nii.gz")
        nib.save(nib.Nifti1Image(evecs, diffusion_affine, header=nifti.header), R"data\dti\evecs.nii.gz")

        # Load results to calculate mean values
        fa_nifti = nib.load(R"data\dti\fa.nii.gz")
        md_nifti = nib.load(R"data\dti\md.nii.gz")
        fa_array = fa_nifti.get_fdata()
        md_array = md_nifti.get_fdata()

        # Print mean FA and MD values
        print("The FA mean is:", np.mean(fa_array))
        print("The MD mean is:", np.mean(md_array))
    except Exception as e:
        print(f"Error during tensor model fitting: {e}")

def diffusion_to_T2(diffusion_data, diffusion_affine, T2_data, T2_affine):
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
    
    
def main(denoising_method="patch2self", use_affine=False):
    """
    Main function to load data, perform motion correction, denoising,
    fit tensor model, apply Gibbs correction, calculate MI scores,
    perform transformation to T2, calculate cross-correlation, and save DTI metrics.

    Parameters:
    - denoising_method: str, "patch2self" or "mppca".
    - use_affine: bool, whether to include the "affine" step in motion correction.
    """
    
    print(f"Starting pipeline with {denoising_method} denoising and {'affine' if use_affine else 'non-affine'} motion correction...")
    diffusion_file_path = R"Data_loaded/3/pdata/1/niiobj_1.nii"
    T2_file_path = R"Data_loaded/2/pdata/1/niiobj_1.nii"
    method_file_path = R"Data_loaded/3/method.npy"
    
    # Define unique save paths for the pipeline
    affine_label = "with_affine" if use_affine else "without_affine"
    denoised_save_path = f"Image_denoised/image_denoised_{denoising_method}_{affine_label}.nii.gz"
    gibbs_save_path = f"Image_denoised/image_gibbs_{denoising_method}_{affine_label}.nii.gz"

    # Load diffusion data
    diffusion_data, diffusion_affine, diffusion_nifti = load_nifti_data(diffusion_file_path, return_img=True)

    # Load T2 data
    T2_data, T2_affine, T2_nifti = load_nifti_data(T2_file_path, return_img=True)

    # Load method file and extract B-values and B-vectors
    bval, bvec = load_method_file(method_file_path)
    gtab = gradient_table(bval, bvec, atol=1)
    bvals = np.array(bval)
 
   

    if bval is not None and bvec is not None:
        print("B-values and B-vectors loaded successfully.")
        
        print("Here they are...")
        
        print("gtab:",gtab)
        print("bvals:", bvals)
        

        # Perform motion correction
        dwi_motion_corrected, reg_affines = perform_motion_correction(
            diffusion_nifti, bval, bvec, diffusion_affine, use_affine=use_affine
        )

        if dwi_motion_corrected is not None:
            dwi_motion_corrected_data = dwi_motion_corrected.get_fdata()

            # Perform denoising based on the specified method
            if denoising_method == "patch2self":
                denoised_data = perform_patch2self_denoising(
                    dwi_motion_corrected_data, bvals, diffusion_nifti, denoised_save_path
                )
            elif denoising_method == "mppca":
                denoised_data = perform_mppca_denoising(
                    dwi_motion_corrected_data, diffusion_nifti, denoised_save_path
                )
            else:
                raise ValueError(f"Unsupported denoising method: {denoising_method}")

            if denoised_data is not None:
                # Perform Gibbs correction
                gibbs_corrected_data = perform_gibbs_correction(
                    denoised_save_path, gibbs_save_path, diffusion_nifti
                )

                # Calculate MI scores
                mi_scores = calculate_mi_scores(
                    diffusion_data,
                    dwi_motion_corrected_data,
                    denoised_data,
                    gibbs_corrected_data
                )

                # Perform Tensor Model Fit
                perform_tensor_model_fit(
                    gibbs_corrected_data, diffusion_affine, diffusion_nifti, gtab
                )

                # Transform diffusion images to T2
                upsampled_diffusion_data, warped_moving = diffusion_to_T2(
                    diffusion_data, diffusion_affine, T2_data, T2_affine
                )

                if upsampled_diffusion_data is not None and warped_moving is not None:
                    print("Transformation and warping to T2 completed successfully.")

                    # Calculate cross-correlation before and after warping
                    cross_corr_value_before = cross_correlation(upsampled_diffusion_data, T2_data)
                    print("Cross-Correlation Value (Before Warping):", cross_corr_value_before)

                    cross_corr_value_after = cross_correlation(warped_moving, T2_data)
                    print("Cross-Correlation Value (After Warping):", cross_corr_value_after)

                    # Combine results into a dictionary
                    return {
                        "mi_scores": mi_scores,
                        "cross_corr_before": cross_corr_value_before,
                        "cross_corr_after": cross_corr_value_after,
                    }

    return None




if __name__ == "__main__":
    results = {}
    
    print("And awayyyy we go!")

    print("Running Patch2Self pipeline without affine...")
    results["patch2self_without_affine"] = main(denoising_method="patch2self", use_affine=False)
    
    print("WHEEL AND COME AGAIN 'airhorn noises'")

    print("\nRunning Patch2Self pipeline with affine...")
    results["patch2self_with_affine"] = main(denoising_method="patch2self", use_affine=True)
    
    print("POOOOOOL UP, WHEEL UP AND COME AGAIN'airhorn noises'")

    print("\nRunning MPPCA pipeline without affine...")
    results["mppca_without_affine"] = main(denoising_method="mppca", use_affine=False)
    
    print("POOOOOOL UP, ONE MORE TIME 'airhorn noises'")

    print("\nRunning MPPCA pipeline with affine...")
    results["mppca_with_affine"] = main(denoising_method="mppca", use_affine=True)

    # If all pipelines completed successfully, plot results
    if all(results.values()):
        steps = ["Raw", "Motion Corrected", "Denoised", "Gibbs Corrected"]

        # Define a list of markers and colors for better visualization
        markers = ["o", "o", "x", "x"]
        colors = ["blue", "green", "orange", "red"]

        # Set the plot size
        plt.figure(figsize=(10, 6))

        for idx, (key, scores) in enumerate(results.items()):
            label = key.replace("_", " ").title()
            plt.plot(
                steps,
                [
                    scores["mi_scores"]["average_mi_before"],
                    scores["mi_scores"]["average_mi_after_motion"],
                    scores["mi_scores"]["average_mi_after_denoise"],
                    scores["mi_scores"]["average_mi_after_gibbs"]
                ],
                label=label,
                marker=markers[idx % len(markers)],  # Cycle through markers
                color=colors[idx % len(colors)],    # Cycle through colors
                linewidth=2,                        # Make lines slightly thicker
                markersize=8                        # Enlarge markers
            )

        # Add gridlines
        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

        # Add labels, title, and legend
        plt.xlabel("Processing Steps", fontsize=12)
        plt.ylabel("Mutual Information (MI)", fontsize=12)
        plt.title("Comparison of MI Scores Across Pipelines", fontsize=14, fontweight="bold")
        plt.legend(loc="upper left", fontsize=10, frameon=True, edgecolor="black", fancybox=True)
        
      

        # Show the plot
        plt.tight_layout()  # Adjust layout to prevent label clipping
        plt.show()
    else:
        print("Error: One or more pipelines failed.")
