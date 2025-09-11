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
import uuid
import shutil
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ast import literal_eval
from monai.visualize import GuidedBackpropSmoothGrad, blend_images
from monai.networks.nets.resnet import ResNet, ResNetBottleneck
from torch import nn
import scipy

def to_subscript(s):
    subscript_map = {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "t": "ₜ"
    }
    return "".join(subscript_map.get(c, c) for c in s)

def string_to_array(string):
    #use literal_eval to convert string to tuple
    array = literal_eval(string)
    return np.array(array) #np.array([int(x) for x in string.replace("(", "").replace(")", "").split(", ")])

def remove_files_from_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
            
# Convert to global coordinates
def convert_local_to_global(image_path, x, y, z):
    itk_image = sitk.ReadImage(image_path)
    x_global, y_global, z_global = itk_image.TransformContinuousIndexToPhysicalPoint((int(x), int(y), int(z)))
    return (x_global, y_global, z_global)

def plot_slice_at_centroid(image, centroid):
    slice_index = centroid[2]
    slice_image = sitk.GetArrayFromImage(image)[slice_index, :, :]
    #clip values between -400, 1400
    
    # Apply lung window settings
    window_width = 1500
    window_level = -600
    min_value = window_level - window_width / 2
    max_value = window_level + window_width / 2

    # Clip the image values to the lung window range
    slice_image = np.clip(slice_image, min_value, max_value)
    # slice_image = np.clip(slice_image, -400, 1400)
    
    #normalize slice image
    slice_image = (slice_image - slice_image.min()) / (slice_image.max() - slice_image.min())
    #plot the slice image
    img_path = f"C:/Users/benit/Desktop/Script/Python/local/demos/nlst/tmp/slice_image.png"
    img_path = f"C:/Users/benit/Desktop/Script/Python/local/demos/nlst/tmp/slice_image_{uuid.uuid4()}.png"
    plt.imshow(slice_image, cmap='gray')
    #create a rectangle around the centroid
    # plt.plot(centroid[0], centroid[1], markersize=0.5)
    plt.gca().add_patch(plt.Rectangle((centroid[0]-40, centroid[1]-40), 80, 80, edgecolor='r', facecolor='none'))
    plt.axis('off')
    plt.savefig(img_path)
    plt.close()
    return img_path

def plot_empy_slice(size = 512):
    img_path = f"C:/Users/benit/Desktop/Script/Python/local/demos/nlst/tmp/slice_image_{uuid.uuid4()}.png"
    plt.imshow(np.zeros((size, size)), cmap='gray')
    plt.axis('off')
    plt.savefig(img_path)
    plt.close()
    return img_path

def get_centroid_input(image):
    #plot the slice image
    img_path = f"C:/Users/benit/Desktop/Script/Python/local/demos/nlst/tmp/slice_input_image_{uuid.uuid4()}.png"
    plt.imshow(image.squeeze()[25,:,:], cmap='gray')
    plt.axis('off')
    plt.savefig(img_path)
    plt.close()
    return img_path

def get_silency_map_fmcib(model, image):
    CHECKPOINT_PATH = "./models/feature_extractor_finetuned.torch"
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    trained_trunk = state_dict["trunk_state_dict"]
    trained_head = state_dict["head_state_dict"]

    trunk = ResNet(
        block=ResNetBottleneck,
        layers=(3, 4, 6, 3),
        block_inplanes=(64, 128, 256, 512),
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=2,
        conv1_t_size=7,
        widen_factor=2,
    )
    trunk.fc = nn.Identity()

    # Add head part of the model
    head0 = nn.Sequential(nn.Linear(4096, 2048, bias=True), nn.ReLU(inplace=True))
    head1 = nn.Sequential(nn.Linear(2048, 512, bias=True), nn.ReLU(inplace=True))
    head2 = nn.Sequential(nn.Linear(512, 256, bias=True), nn.ReLU(inplace=True))
    head3 = nn.Sequential(nn.Linear(256, 2, bias=True))
    head = nn.Sequential(head0, head1, head2, head3)

    trunk.load_state_dict(trained_trunk)
    head.load_state_dict(trained_head)
    model = nn.Sequential(trunk, head)
    model.eval()

    activation_model = GuidedBackpropSmoothGrad(model)
    
    out = activation_model(image)

    # Apply Gaussian smoothing to the output and normalize
    smooth_out = scipy.ndimage.gaussian_filter(out.cpu().squeeze().detach().numpy(), sigma=1)
    smooth_out = (smooth_out - smooth_out.min()) / (smooth_out.max() - smooth_out.min())

    # Blend the original image with the activation map
    blended = blend_images(
        image,
        torch.from_numpy(smooth_out).unsqueeze(0).unsqueeze(0),
        alpha=0.6,
        cmap="jet",
        rescale_arrays=False,
        transparent_background=False,
    )

    saliency_path = f"./tmp/saliency_{uuid.uuid4()}.png"
    plt.figure()
    plt.imshow(blended.squeeze()[:,25,:,:].permute(1,2,0))
    plt.axis('off')
    plt.savefig(saliency_path)
    plt.close()
    return saliency_path

def extract_fmcib_features(image, centroid):
    #save image to temp folder
    image_path = "./tmp/image.nrrd"
    sitk.WriteImage(image, image_path)
    
    #create a csv with image path and coordinates
    csv_df = pd.DataFrame({"image_path": [image_path], "coordX_orig": [centroid[0]], "coordY_orig": [centroid[1]], "coordZ_orig": [centroid[2]]})
    csv_df['coordX'], csv_df['coordY'], csv_df['coordZ'] = zip(*csv_df.apply(lambda x: convert_local_to_global(x['image_path'], x['coordX_orig'], x['coordY_orig'], x['coordZ_orig']), axis=1))
    csv_path = "./tmp/tmp.csv"
    csv_df.to_csv(csv_path, index=False)
    
    # Extract features from FMCIB
    dataloader = get_dataloader(csv_path, spatial_size=[50,50,50], precropped=False, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trunk = resnet50(
        pretrained=False,
        n_input_channels=1,
        widen_factor=2,
        conv1_t_stride=2,
        feed_forward=False,
        bias_downsample=True,
        )
    
    model = LoadModel(trunk=trunk, weights_path='./models/feature_extractor_model_weights.torch').to(device)

    feature_list = []
    
    model.eval()
    image_list = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        image_list.append(batch)
        feature = model(batch.to(device)).detach().cpu().numpy()
        feature_list.append(feature)

    features = np.concatenate(feature_list, axis=0)
    # Flatten features into a list
    features = features.reshape(-1, 4096)

    # Add the features to the dataframe
    features = pd.DataFrame(features, columns=[f"pred_{idx}" for idx in range(4096)])
    
    #plot the slice image
    slice_image = get_centroid_input(image_list[0])
    
    ## saliency map
    saliency_path = get_silency_map_fmcib(model, image_list[0])
    
    #remove temp files
    os.remove(image_path)
    os.remove(csv_path)
    
    return features, slice_image, saliency_path