# Import necessary libraries
import gradio as gr
import numpy as np
from utils import plot_slice_at_centroid, extract_fmcib_features, plot_empy_slice, remove_files_from_folder, string_to_array, to_subscript
from process_linear import predict_linear
from process_spatio_temporal import spatiotemporal_classifier, show_attention_weights_bars
import SimpleITK as sitk
import os

summary_text_markdown = """

# Lung Nodule Malignancy Prediction Tool

This tool predicts the probability of a lung nodule being malignant using a **spatio-temporal model**. The model analyzes up to three 3D CT scans of the same nodule taken at different timepoints. 
The prediction considers the **temporal progression** of the nodule, if multiple timepoints are available.

#### Inputs: 
- **CT Scans**: Provide up to three 3D CT scans of the same lung nodule taken at different timepoints: 
  - **<i>T<sub>t</sub></i>**: The most recent and last available timepoint.
  - **<i>T<sub>t-1</sub></i>**: The previous timepoint (optional).
  - **<i>T<sub>t-2</sub></i>**: The earliest timepoint (optional).
- **Nodule Centroids**: Provide the coordinates of the nodule for each available CT scan in the format `(x, y, z)`.

#### Outputs:
- **Malignancy Probability**: The likelihood that the nodule is malignant, based on the provided data. If multiple timepoints are available, the prediction considers the temporal progression of the nodule.
- **Temporal Attention Weights**: Highlights the importance of each timepoint \(<i>T<sub>t-2</sub></i>, <i>T<sub>t-1</sub></i>, <i>T<sub>t</sub></i>\) in the prediction, if more than one timepoint is given.
- **Silency Maps**: Visualize the silency maps for each timepoint to understand the model's focus areas.

#### How to Use This Tool:
1. **Upload Data**:
   - Provide up to three 3D CT scans in NRRD format, each corresponding to a specific timepoint (<i>T<sub>t-2</sub></i>, <i>T<sub>t-1</sub></i>, <i>T<sub>t</sub></i>).
   - You can provide **complete data** (all three timepoints) or **partial data** (one or two timepoints). 
   - Enter the nodule centroid coordinates for each available scan in the format: `(x, y, z)`.
2. **Run the Prediction**:
   - Click the **Classify Nodule Malignancy** button.
   - View the malignancy probability and interpret the attention weights.
3. **Test with Examples**:
   - Use the preloaded examples to explore how the model works with complete or partial data.
"""

spacer = """



"""

def reset_outputs(malignancy_output, box1_output, box2_output, box3_output, box4_output, box5_output, box6_output, box7_output, box8_output, box9_output, attention_weights, attention_output):
    return None, None, None, None, None, None, None, None, None, None, None, None

def reset_and_remove(*args):
        remove_files_from_folder("./tmp")
        return reset_outputs(*args)
    
# Create a Gradio interface
with gr.Blocks(css="style.css") as demo:
    
    # Header
    gr.HTML(
        value=f"<center><font size='6'><bold> BIT-UPM Spatio-temporal Nodule Malignancy Prediction</bold></font></center>"
    )
        
    #add a table where in the first row first column you have the image logo 1, second row first column image logo 2, third row first column image logo 3; 
    # then in the second column a text for explaining the demo
    logo_upm = './logos/Logo UPM.png'
    logo_ciber = './logos/Logo CIBER BBN.jpg'
    logo_bit = './logos/bit_logo.png'
    with gr.Row():
        gr.Image(logo_upm, elem_id="logo_upm", height=200, width=200)
        gr.Image(logo_ciber, elem_id="logo_ciber", height=200, width=200)
        gr.Image(logo_bit, elem_id="bit_logo", height=200, width=200)
        
    with gr.Row():
        gr.Markdown(value=summary_text_markdown, elem_classes="summary_text")
        
    # Input components
    with gr.Row():
        image1 = gr.File(label=f"Upload NRRD CT  T{to_subscript('t-2')}")
        image2 = gr.File(label=f"Upload NRRD CT  T{to_subscript('t-1')}")
        image3 = gr.File(label=f"Upload NRRD CT  T{to_subscript('t')}")
    
    with gr.Row():
        centroid1 = gr.Textbox(label=f"Centroid  T{to_subscript('t-2')} (x, y, z)", placeholder="e.g., (100, 150, 125)")
        centroid2 = gr.Textbox(label=f"Centroid  T{to_subscript('t-1')} (x, y, z)", placeholder="e.g., (120, 160, 125)")
        centroid3 = gr.Textbox(label=f"Centroid T{to_subscript('t')} (x, y, z)", placeholder="e.g., (110, 140, 125)")
    
    # Button to trigger the classification
    classify_button = gr.Button("Classify Nodule Malignancy", size = "lg", elem_classes=["custom-button"])
    
    gr.Markdown(value = spacer) 
    
    with gr.Row():
        box1_output = gr.Image(label=f"CT T{to_subscript('t-2')} Slice at Centroid")
        box2_output = gr.Image(label=f"CT T{to_subscript('t-1')} Slice at Centroid")
        box3_output = gr.Image(label=f"CT T{to_subscript('t')} Slice at Centroid")
    
    with gr.Row():
        box4_output = gr.Image(label=f"Input Centroid Slice T{to_subscript('t-2')}")
        box5_output = gr.Image(label=f"Input Centroid Slice T{to_subscript('t-1')}")
        box6_output = gr.Image(label=f"Input Centroid Slice T{to_subscript('t')}")
        
    with gr.Row():
        box7_output = gr.Image(label=f"Silency Map Centroid Slice T{to_subscript('t-2')}")
        box8_output = gr.Image(label=f"Silency Map Centroid Slice T{to_subscript('t-1')}")
        box9_output = gr.Image(label=f"Silency Map Centroid Slice T{to_subscript('t')}")
        
    # Output components in separate rows
    with gr.Row():
        malignancy_output = gr.Label(label="Lung Nodule Malignancy Probability", elem_classes=['custom_label'])
        # malignancy_output = gr.Textbox(label="Nodule Malignancy Probability", elem_id="malignancy_output", show_label=True, scale = 2, show_copy_button=True))
    
    #create a new row for the attention weights: consider that the attention weights are a list of 3 elements (one for each timepoint)
    with gr.Row(): 
        attention_weights = gr.Label(label="Attention weights")
        attention_output = gr.Plot(label="Attention Weights Visualization")  # Plot output
        # attention_button = gr.Button("Display Attention Weights", size = "lg")
    
    # Preloaded examples
    examples_dir = ".//examples/"
    gr.Examples(
        examples=[
                    [
                        f"{examples_dir}122117_T0.nrrd", 
                        f"{examples_dir}122117_T1.nrrd", 
                        f"{examples_dir}122117_T2.nrrd",
                        "(138, 244, 114)", 
                        "(157, 266, 102)",
                        "(160, 258, 104)", 
                    ],
                    [
                        f"{examples_dir}100560_T0.nrrd", 
                        f"{examples_dir}100560_T1.nrrd", 
                        f"{examples_dir}100560_T2.nrrd",
                        "(365, 332, 117)", 
                        "(371, 334, 107)",
                        "(370, 333, 115)", 
                    ],
                    [
                        f"{examples_dir}109198_T0.nrrd", 
                        f"{examples_dir}109198_T1.nrrd", 
                        f"{examples_dir}109198_T2.nrrd",
                        "(128, 194, 111)", 
                        "(126, 185, 128)",
                        "(120, 160, 110)", 
                    ],
                    [
                        f"{examples_dir}116176_T0.nrrd", 
                        f"{examples_dir}116176_T1.nrrd", 
                        f"{examples_dir}116176_T2.nrrd",
                        "(433, 269, 149)", 
                        "(427, 256, 75)",
                        "(440, 233, 66)", 
                    ],
                    [
                        f"{examples_dir}100095_T0.nrrd", 
                        f"{examples_dir}100095_T1.nrrd", 
                        f"{examples_dir}100095_T2.nrrd",
                        "(334, 354, 120)", 
                        "(325, 338, 128)",
                        "(334, 318, 131)", 
                    ],
                    [
                        f"{examples_dir}100113_T0.nrrd", 
                        f"{examples_dir}100113_T1.nrrd", 
                        f"{examples_dir}100113_T2.nrrd",
                        "(111, 162, 32)", 
                        "(125, 154, 70)",
                        "(110, 125, 64)", 
                    ],
                ],
        inputs=[image1, image2, image3, centroid1, centroid2, centroid3],
        cache_examples=False,
        label="Preloaded Examples",
    )
    
    # Define the event listener for the button
    classify_button.click(
        spatiotemporal_classifier,
        inputs=[image1, image2, image3, centroid1, centroid2, centroid3],
        outputs=[malignancy_output, box1_output, box2_output, box3_output, box4_output, box5_output, box6_output, box7_output, box8_output, box9_output, attention_weights, attention_output],
    )

    # Reset button for clearing outputs
    reset_button = gr.Button("Reset Outputs")
    reset_button.click(
        reset_and_remove,
        inputs=[malignancy_output, box1_output, box2_output, box3_output, box4_output, box5_output, box6_output, box7_output, box8_output, box9_output, attention_weights, attention_output],
        outputs=[malignancy_output, box1_output, box2_output, box3_output, box4_output, box5_output, box6_output, box7_output, box8_output, box9_output, attention_weights, attention_output]
    )
    
if __name__ == "__main__":
    demo.launch()
    
