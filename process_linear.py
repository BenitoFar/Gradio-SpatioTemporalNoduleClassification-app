import nrrd
from fmcib.run import get_features
from fmcib.preprocessing import get_transforms, get_dataloader
from fmcib.visualization import visualize_seed_point
from fmcib.models import LoadModel
import monai
import SimpleITK as sitk
import pandas as pd
import joblib
import os
import torch 
from monai.networks.nets import resnet50
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import plot_slice_at_centroid, extract_fmcib_features, string_to_array

def predict_linear(features):
    #import scaler
    scaler = joblib.load("./preprocessing/scaler_linear_model.pkl")
    features = scaler.transform(features)
    
    #import linear model
    model = joblib.load("./models/linear_model.pkl")
    
    #run model
    probs = model.predict_proba(features)
    
    return probs

# Gradio interface function
def linear_classifier(nrrd_file, centroid):
    centroid = string_to_array(centroid)
    
    labels = ['Benign Nodule', 'Malignant Nodule']
    # Load the NRRD file
    image=sitk.ReadImage(nrrd_file)

    features, slice_input_image = extract_fmcib_features(image, centroid)
    
    # Make prediction
    malignancy_prob = predict_linear(features)
    confidences = {labels[i]: float(malignancy_prob.squeeze()[i]) for i in range(len(labels))}
    
    # Get the slice image
    slice_image = plot_slice_at_centroid(image, centroid)
    
    return confidences, slice_image, slice_input_image

if __name__ == "__main__":
    """
    Run on a test case
    """
    # Example inputs
    example_ct_file = "100012_T1_1.2.840.113654.2.55.50761756412482430061802871163319122196.nrrd"  # Replace with the path to your example NRRD file
    example_coordinates = "(165, 329, 119)"  # Example centroid coordinates
    
    confidences, slice_image, slice_input_image =linear_classifier(example_ct_file, example_coordinates)
    print(confidences)