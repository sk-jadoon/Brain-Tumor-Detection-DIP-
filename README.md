# Brain-Tumor-Detection-DIP-
# Brain Tumor Detection using Digital Image Processing (DIP)

## Project Overview

This project implements a Digital Image Processing (DIP) pipeline to detect and segment brain tumors from MRI images. The system enhances medical images, extracts tumor regions, and evaluates results using quantitative metrics.

---

## Objectives

* Improve MRI image quality using preprocessing techniques
* Segment tumor regions from brain images
* Extract features such as tumor area
* Evaluate performance on a test dataset
* Provide reproducible and structured results

---

##Dataset

The dataset consists of brain MRI images divided into two categories:

* **yes/** → Images with tumor
* **no/** → Images without tumor

> Note: Ensure the dataset path is correctly set in the code.

Example structure:

```
dataset/
   yes/
   no/
```

---

## ⚙️ Tools & Libraries

The project is implemented in Python using the following libraries:

* OpenCV
* NumPy
* Matplotlib
* Pandas
* scikit-image
* scikit-learn

---

## Methodology

### 1. Image Acquisition

MRI images are loaded from the dataset directory.

### 2. Preprocessing

* Gaussian Blur (noise reduction)
* Histogram Equalization (contrast enhancement)

### 3. Segmentation

* Thresholding to separate tumor region
* Morphological operations (erosion + dilation) to remove noise

### 4. Tumor Detection

* Contour detection is used to identify the tumor region
* Largest contour is assumed as tumor

### 5. Feature Extraction

* Tumor area is calculated using contour area

### 6. Classification (Basic Rule-Based)

* If area > threshold → Tumor
* Otherwise → No Tumor

---

##  Test Set

A total of **20 images** are selected:

* 10 tumor images
* 10 non-tumor images

These images are processed and evaluated.

---

##  Evaluation Metrics

The following metrics are used:

* **Accuracy**
* **Precision**
* **Recall**
* **Dice Coefficient**

Results are stored in:

```
results/evaluation.csv
```

---

##  Project Structure

```
BrainTumorProject/
│
├── dataset/
│   ├── yes/
│   └── no/
│
├── test_images/
├── results/
│   └── evaluation.csv
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

##  How to Run

### 1. Install dependencies

```
pip install opencv-python numpy matplotlib pandas scikit-image scikit-learn
```

### 2. Set dataset path in notebook

```
dataset_path = r"F:\Projects\Brain Tumor Detection\archive"
```

(Adjust path according to your system)

### 3. Run notebook

Execute all cells in `notebook.ipynb`

---

##  Common Errors & Fixes

### FileNotFoundError

* Ensure dataset path is correct
* Verify folder names (yes/no)
* Check for extra nested folders

### Image Not Loading

* Confirm file path is valid
* Check image format (.jpg, .png)

---

##  Results

* Tumor regions successfully detected using DIP techniques
* Performance evaluated on 20 test images
* Results stored in CSV file

---

##  Limitations

* Threshold-based segmentation may fail on low-contrast images
* Assumes largest contour is tumor
* Not suitable for highly complex MRI scans

---

## Future Improvements

* Use K-means clustering for better segmentation
* Implement CNN for higher accuracy
* Develop GUI for user interaction

---

##  Author

Sidra Jadoon

---

## References

* Gonzalez & Woods, *Digital Image Processing*
* OpenCV Documentation: https://docs.opencv.org/
* scikit-image Documentation: https://scikit-image.org/docs/
* Kaggle MRI Dataset

---
