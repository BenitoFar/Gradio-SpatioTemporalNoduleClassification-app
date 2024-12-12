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
from utils import plot_slice_at_centroid, extract_fmcib_features, string_to_array, plot_empy_slice, remove_files_from_folder, to_subscript
from torch.utils.data import DataLoader, TensorDataset
import pickle
import io
from common import RNNClassifier
from matplotlib import cm
import matplotlib.pyplot as plt

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: 
            return super().find_class(module, name)

# Function to display attention weights with color
def show_attention_weights(attention_weights):
    
    # Create a display with color-coded attention weights
    labels = ["Tₜ₋₂", "Tₜ₋₁", "Tₜ"]
    colored_labels = [f"<span style='color:{weight_to_color(w)}'>{label}: {w:.2f}</span>" 
                      for label, w in zip(labels, attention_weights)]
    
    # Return the colored labels as a formatted string
    return "<br>".join(colored_labels)
 
# Function to map weight to color
def weight_to_color(weight):
    # Use a color map to convert weight to a color (blue -> red)
    cmap = cm.get_cmap('coolwarm')
    rgba = cmap(weight)  # Convert weight to RGBA
    return f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"

def show_attention_weights_bars(attention_weights):
    labels = ["Tₜ₋₂", "Tₜ₋₁", "Tₜ"]
    colors = [cm.coolwarm(w) for w in attention_weights]
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh(labels, attention_weights, color=colors)  # Color bars based on weights
    ax.set_xlim(0, 1)
    ax.set_xlabel('Attention Weight')
    # ax.set_title('Attention Weights Over Time')
    # Return the plot as an image
    return fig
   
def predict_spatiotemporal(features):
    #Applying scaling only to non-NaN values
    scaler = joblib.load("./preprocessing/scaler_spatiotemp_model.pkl")
    reshaped_features = features.reshape(-1, features.shape[-1])
     # Create a mask for non-NaN values
    mask = ~np.isnan(reshaped_features)
    reshaped_features[mask] = scaler.transform(reshaped_features[mask].reshape(-1, features.shape[-1])).flatten()
    features = reshaped_features.reshape(features.shape)
    
    #fill nan values with constant value = -1
    features[np.isnan(features)] = -1
    
    #import linear model
    with open("./models/models_globattcrnn.pkl", 'rb') as pickle_file:
        models = CPU_Unpickler(pickle_file).load()
    
    if not isinstance(models, list):
        models = [models]
        
    probs = np.zeros((len(features), ))
    attention_weights = [np.zeros((len(features), 3)) for _ in range(len(models))]
    
    # Evaluate the classifier on the test set
    test_dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, classifier in enumerate(models):
            fold_probs = []
            att_w = []
            for batch_X in test_loader:
                # batch_X = batch_X.to(device)
                outputs, attn_weights = classifier(batch_X[0])
                if attn_weights is not None:
                    att_w.extend(attn_weights.squeeze().cpu().numpy())
                fold_probs.append(outputs.squeeze().cpu().numpy())
            attention_weights[i] = np.array(att_w)
            probs += np.array(fold_probs)
    probs /= len(models)
    attention_weights = attention_weights[0]
    # attention_weights = [att_w / len(models) for att_w in attention_weights]
    
    if probs.ndim == 1:
        probs = np.stack([1 - probs, probs], axis=1)
    return probs, attention_weights

# Gradio interface function
def spatiotemporal_classifier(image1_path, image2_path, image3_path, centroid1, centroid2, centroid3):
     #check if at least one image is provided and the corresponding centroid
    if not image1_path and not image2_path and not image3_path:
        return "Please upload at least one NRRD CT scan"
    
    if not centroid1 and not centroid2 and not centroid3:
        return "Please provide at least one centroid"
    
    # Read the images if they are provided
    if image1_path:
        centroid1 = string_to_array(centroid1)
        image1 = sitk.ReadImage(image1_path)
        features1, slice_input_1, gradcam1 = extract_fmcib_features(image1, centroid1)
        slice_image_1 = plot_slice_at_centroid(image1, centroid1)
    else:
        image1 = None
        centroid1 = np.full((1, 3), np.nan)
        features1 = np.full((1, 4096), np.nan)
        slice_image_1 = plot_empy_slice()
        slice_input_1 = plot_empy_slice(size = 50)
        gradcam1 = plot_empy_slice(size = 50)
        
    if image2_path:
        centroid2 = string_to_array(centroid2)
        image2 = sitk.ReadImage(image2_path)
        features2, slice_input_2, gradcam2 = extract_fmcib_features(image2, centroid2)
        slice_image_2 = plot_slice_at_centroid(image2, centroid2)
    else:
        image2 = None
        centroid2 = np.full((1, 3), np.nan)
        features2 = np.full((1, 4096), np.nan)
        slice_image_2 =  plot_empy_slice()
        slice_input_2 = plot_empy_slice(size = 50)
        gradcam2 = plot_empy_slice(size = 50)
        
    if image3_path:
        centroid3 = string_to_array(centroid3)
        image3 = sitk.ReadImage(image3_path)
        features3, slice_input_3, gradcam3 = extract_fmcib_features(image3, centroid3)
        slice_image_3 = plot_slice_at_centroid(image3, centroid3)
    else:
        image3 = None
        centroid3 = np.full((1, 3), np.nan)
        features3 = np.full((1, 4096), np.nan)
        slice_image_3 =  plot_empy_slice()
        slice_input_3 = plot_empy_slice(size = 50)
        gradcam3 = plot_empy_slice(size = 50)
        
    #concatenate features
    features = np.concatenate([features1, features2, features3], axis=0)[np.newaxis, :]
    
    labels = ['Benign Nodule', 'Malignant Nodule']
    
    # Make prediction
    malignancy_prob, attention_weights = predict_spatiotemporal(features)
    confidences = {labels[i]: float(malignancy_prob.squeeze()[i]) for i in range(len(labels))}
    attention_weights_dict = {f"Most relevant timestep = T{(to_subscript('t-' + str(2-i)) if (2-i) != 0 else 't')}": attention_weights[i] for i in range(3)}
    attention_weights_plot = show_attention_weights_bars(attention_weights)
    # Return the malignancy score and the extracted boxes
    return confidences, slice_image_1, slice_image_2, slice_image_3, slice_input_1, slice_input_2, slice_input_3, gradcam1, gradcam2, gradcam3, attention_weights_dict, attention_weights_plot

if __name__ == "__main__":
    """
    Run on a test case
    """
    examples_dir = ".//examples/"
    example =[
            f"{examples_dir}100560_T0_1.2.840.113654.2.55.175786539281779995340651740089966349113.nrrd", 
            f"{examples_dir}100560_T1_1.2.840.113654.2.55.197315462131770413312066991296742720079.nrrd", 
            f"{examples_dir}100560_T2_1.2.840.113654.2.55.45527388302862716797347470007986763169.nrrd",
            "(365, 332, 117)", 
            "(371, 334, 107)",
            "(370, 333, 115)"]
            
    
    confidences, slice_image_1, slice_image_2, slice_image_3, attention_weights = spatiotemporal_classifier(
        example[0], example[1], example[2], example[3], example[4], example[5]
    )
    print(confidences)