
# VisionPro

This project is a simple alternative to the project I uploaded **Fingerprint scanner VAE** which provides results efficiently. It involves preprocessing, applying edge filters, generating composite images, extracting SIFT features, and matching test samples using feature similarity.

---

## Project Highlights

- Converts `.bmp` files to `.jpg`
- Resizes images to a standard resolution
- Applies **Sobel**, **Canny**, and **Laplacian** edge filters
- Generates composite images by merging filtered outputs
- Extracts **SIFT features** for each image
- Performs **FLANN-based feature matching** with a test image
- Saves image metadata in CSV
- Visualizes and saves keypoints for inspection

---

## Directory Structure

```
project_root/
├── bmp_images/             # Original dataset in BMP format (Not present here)
├── jpg_images/             # Converted JPEGs (Not present here)
├── resized_images/         # 256x256 resized images (Not present here)
├── sobel_images/           # Sobel-filtered outputs (Not present here)
├── canny_images/           # Canny-filtered outputs (Not present here)
├── laplacian_images/       # Laplacian-filtered outputs (Not present here)
├── composite_images/       # Combined edge-filter images (Not present here)
├── test_images/            # Folder with a single test image (test.jpg) (Not present here)
├── image_dimensions.csv    # CSV with dimensions of processed images (will be auto-generated)
├── VisionPro.py            # Main pipeline script
```

---

##  How to Use

### 1. Organize Your Dataset

- Place your `.bmp` images in `bmp_images/`
- Ensure your test image is named `test.jpg` and located in `test_images/`

### 2. Update Folder Paths

Edit the top of `VisionPro.py` to match your folder names:

```python
input_folder = 'bmp_images'
output_folder = 'jpg_images'
path3 = 'resized_images'
Sobel = 'sobel_images'
Canny = 'canny_images'
Laplacian = 'laplacian_images'
Composite = 'composite_images'
Test = 'test_images'
csv_file = 'image_dimensions.csv'
```

### 3. Install Dependencies

```bash
pip install opencv-python numpy pillow
```

---

### 4. Run the Pipeline

```bash
python LIFT_Pipeline_Cleaned.py
```

---

## Output & Logs

- Transformed images saved in their respective folders
- Composite images represent the maximum value from edge filters
- Images with SIFT keypoints are saved with `kp_` prefix
- Best matching image for the test sample is printed in the terminal
- Runtime performance is displayed after completion

---

## Techniques Used

- **OpenCV** for image processing and filters
- **SIFT (Scale-Invariant Feature Transform)** for feature detection
- **FLANN (Fast Library for Approximate Nearest Neighbors)** for matching
- **NumPy** and **PIL** for matrix operations and metadata

