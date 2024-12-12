import gradio as gr
import nrrd
from fmcib.run import get_features
from fmcib.preprocessing import get_transforms, get_dataloader
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
from process_linear import extract_fmcib_features, predict_linear, plot_slice_at_centroid, linear_classifier

# Example inputs
example_ct_file1 = "100012_T0_1.2.840.113654.2.55.240231128564881525363489796879328810792.nrrd"  # Replace with the path to your example NRRD file
example_ct_file1_coordinates = "(199, 313, 126)"  # Example centroid coordinates

example_ct_file2 = "100012_T1_1.2.840.113654.2.55.50761756412482430061802871163319122196.nrrd"  # Replace with the path to your example NRRD file
example_ct_file2_coordinates = "(165, 329, 119)"  # Example centroid coordinates

# Gradio layout
description = "Upload a 3D NRRD CT scan and provide nodule centroid coordinates (cx, cy, cz) to predict malignancy probability."

# Update Gradio layout
interface = gr.Interface(
    fn=linear_classifier,
    inputs=[
        gr.File(label="Upload NRRD CT Image"),
        gr.Textbox(lines=1, placeholder="Enter nodule coordinates as (cx, cy, cz)", label="Nodule Coordinates"),
    ],
    outputs=[
        gr.Label(label="Malignancy Probability"),
        gr.Image(label="CT Slice at Centroid"),
        gr.Image(label="Model Input at Centroid")
    ],
    title="Lung Nodule Malignancy Probability",
    description=description,
    examples=[
        [example_ct_file1, example_ct_file1_coordinates],
        [example_ct_file2, example_ct_file2_coordinates] 
    ]
)

interface.launch(debug=True)