import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy


# --- STEP 1: AUTOMATIC PATH DISCOVERY ---
def find_image_folder(start_path):
    print(f"Searching for images in: {start_path}...")
    for root, dirs, files in os.walk(start_path):
        # Look for folders named 'images' that actually contain files
        if 'images' in root.lower() and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            return root
    return None


# Start searching from your Project root
base_search_path = r'F:\Projects\Brain Tumor Detection'
image_folder = find_image_folder(base_search_path)

if image_folder is None:
    print("Critical Error: Could not find any folder containing images.")
    print("Please check if your F: drive is plugged in properly.")
    exit()
else:
    print(f" Found images at: {image_folder}")


# --- STEP 2: IMAGE PROCESSING PIPELINE ---
def enhance_mri(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None, None

    # Noise Reduction
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Segmentation (Otsu Thresholding)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, enhanced, mask


# --- STEP 3: PROCESS 20 IMAGES ---
all_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
test_set = all_files[:20]

results = []
for name in test_set:
    path = os.path.join(image_folder, name)
    orig, enh, mask = enhance_mri(path)

    if orig is not None:
        results.append({
            "Image_ID": name,
            "Entropy": round(shannon_entropy(enh), 4),
            "Tumor_Area_Pixels": np.sum(mask == 255)
        })

# Save the Quantitative Report
df = pd.DataFrame(results)
df.to_csv("DIP_Final_Report.csv", index=False)
print("\n--- Quantitative Report Preview ---")
print(df.head())

# --- STEP 4: VISUALIZATION ---
if len(test_set) > 0:
    sample_path = os.path.join(image_folder, test_set[0])
    o, e, m = enhance_mri(sample_path)
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(o, cmap='gray'), plt.title("Original")
    plt.subplot(132), plt.imshow(e, cmap='gray'), plt.title("Enhanced (CLAHE)")
    plt.subplot(133), plt.imshow(m, cmap='gray'), plt.title("Segmented")
    plt.show()