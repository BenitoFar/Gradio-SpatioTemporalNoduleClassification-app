# Spatio-Temporal AI for Lung Cancer Screening Nodule Assessment - GRADIO APP

Predict the probability of a lung nodule being **malignant** using a **spatio-temporal deep learning model**.  
The model analyzes **up to three longitudinal 3D CT scans** of the same nodule and considers its **temporal progression**.

**Implementation details:**  
- The **feature extractor (encoder)** is based on the approach from [Pai et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957482/).  
- The **temporal model and attention module** are based on [Farina et al., 2025](https://pubmed.ncbi.nlm.nih.gov/40818205/), with improvements for longitudinal nodule analysis.


[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) 
[![Gradio](https://img.shields.io/badge/Gradio-3.0-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Demo Video

[▶️ Watch the demo video](assets/demo_compressed.mp4)

---

## Features

- Analyze **up to 3 timepoints** of the same lung nodule.
- Compute **malignancy probability**.
- Provide **temporal attention weights** to understand model focus.
- Generate **saliency maps** to visualize regions influencing the prediction.
- Accept **partial or complete longitudinal data**.

---

## Inputs

1. **CT Scans (NRRD format)**  
   - `T_t`: Most recent timepoint (mandatory).  
   - `T_t-1`: Previous timepoint (optional).  
   - `T_t-2`: Earliest timepoint (optional).

2. **Nodule Centroids**  
   - Coordinates for each scan in the format `(x, y, z)`.

---

## Outputs

- **Malignancy Probability**: Likelihood that the nodule is malignant.
- **Temporal Attention Weights**: Shows which timepoints contribute most to the prediction.
- **Saliency Maps**: Visualize the regions in the CT scans that influenced the prediction.

---

## Installation

Clone the repository and install dependencies:

```bash
git https://github.com/BenitoFar/Gradio-SpatioTemporalNoduleClassification-app.git
cd Gradio-SpatioTemporalNoduleClassification-app
pip install -r requirements.txt 
```
---

## Usage

Run the main Gradio app script:
```bash
python gradio_spatio-temporal_model.py 
```

This will launch a local Gradio interface where you can:

- Upload up to three 3D CT scans (NRRD format)
- Enter the nodule centroid coordinates for each scan `(x, y, z)`
- Click **Classify Nodule Malignancy**
- View results:
  - Malignancy probability
  - Temporal attention weights
  - Saliency maps

---

## Notes

- The app also supports partial longitudinal input if full data is unavailable.
- Make sure to have the `models/` and `preprocessing/` folders with the required trained weights in the repository.
  (These can be accessed upon request.)

---

## References

1. **Spatio-temporal deep learning with temporal attention for indeterminate lung nodule classification**  
   Farina, B., Carbajo Benito, R., Montalvo-García, D., Bermejo Peláez, D., Seijo Maceiras, L., & Ledesma Carbayo, M.J. (2025). *Computers in Biology and Medicine*.  
   [Link to paper](https://pubmed.ncbi.nlm.nih.gov/40818205/)

2. **Foundation model for cancer imaging biomarkers**  
   Pai, S., Bontempi, D., Hadzic, I., Prudente, V., Sokač, M., Chaunzwa, T.L., Bernatz, S., Hosny, A., Mak, R.H., Birkbak, N.J., & Aerts, H.J.W.L. (2024). *Nature Machine Intelligence*.  
   [Link to paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957482/)

---

## License

This project is licensed under the [MIT License](LICENSE)
