
import os
import cv2
import numpy as np
import csv
from PIL import Image
import time

start_time = time.time()

# Define all folder paths here
input_folder = 'path_to_bmp'
output_folder = 'path_to_jpg'

path1 = output_folder  # Source for further processing (JPGs)
path3 = 'resized_images'
Sobel = 'sobel_images'
Canny = 'canny_images'
Laplacian = 'laplacian_images'
Composite = 'composite_images'
Test = 'test_images'
csv_file = 'image_dimensions.csv'

# Create directories if they don't exist
for path in [output_folder, path3, Sobel, Canny, Laplacian, Composite, Test]:
    os.makedirs(path, exist_ok=True)

# Convert .bmp to .jpg
bmp_files = [file for file in os.listdir(input_folder) if file.endswith('.bmp')]
for idx, bmp_file in enumerate(bmp_files, 1):
    image_bmp = cv2.imread(os.path.join(input_folder, bmp_file))
    output_file = f"{idx}.jpg"
    cv2.imwrite(os.path.join(output_folder, output_file), image_bmp, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# Resize, Edge Filters
img_files = sorted([f for f in os.listdir(path1) if f.lower().endswith('.jpg')])

for img_file in img_files:
    img_path = os.path.join(path1, img_file)
    img = cv2.imread(img_path)
    resize = cv2.resize(img, (256, 256))
    cv2.imwrite(os.path.join(path3, img_file), resize)

    # Sobel
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite(os.path.join(Sobel, img_file), sobel)

    # Canny
    canny = cv2.Canny(gray, 50, 120) # these calues are NOT original which were used in final implementation
    cv2.imwrite(os.path.join(Canny, img_file), canny)

    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_16S)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite(os.path.join(Laplacian, img_file), laplacian)


common_files = list(set(os.listdir(Sobel)) & set(os.listdir(Canny)) & set(os.listdir(Laplacian)))

for fname in common_files:
    merged_images = []
    for folder in [Sobel, Canny, Laplacian]:
        img = cv2.imread(os.path.join(folder, fname))
        merged_images.append(img)

    composite_image = np.maximum.reduce(merged_images)
    cv2.imwrite(os.path.join(Composite, fname), composite_image)

# Save Image Dimensions to CSV
data = [['Image Name', 'Width', 'Height']]
for filename in os.listdir(Composite):
    if filename.lower().endswith('.jpg'):
        with Image.open(os.path.join(Composite, filename)) as img:
            data.append([filename, img.width, img.height])

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Extract SIFT Features and Draw Keypoints
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=12)
for filename in os.listdir(Composite):
    img_path = os.path.join(Composite, filename)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    kp, des = sift.detectAndCompute(gray_filtered, None)
    img_with_kp = cv2.drawKeypoints(img, kp, None)
    cv2.imwrite(os.path.join(Composite, f'kp_{filename}'), img_with_kp)

# Match Test Image Against Database Using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

test_img_path = os.path.join(Test, "test.jpg")  # Ensure this file exists
if os.path.exists(test_img_path):
    test_img = cv2.imread(test_img_path)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_kp, test_des = sift.detectAndCompute(test_gray, None)
    test_des /= (test_des.sum(axis=1, keepdims=True) + 1e-7)

    best_match = None
    best_match_count = 0

    for filename in os.listdir(Composite):
        if not filename.startswith('kp_') and filename.lower().endswith('.jpg'):
            db_img_path = os.path.join(Composite, filename)
            db_img = cv2.imread(db_img_path)
            db_gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
            db_kp, db_des = sift.detectAndCompute(db_gray, None)

            if db_des is not None and len(db_des) >= 2:
                db_des /= (db_des.sum(axis=1, keepdims=True) + 1e-7)
                matches = flann.knnMatch(test_des, db_des, k=2)

                good_matches = [m for m, n in matches if m.distance < 0.95 * n.distance]

                if len(good_matches) > best_match_count:
                    best_match = filename
                    best_match_count = len(good_matches)

    print(f"Best match: {best_match}")
else:
    print("Test image not found. Please place 'test.jpg' in the Test folder.")

print("Time elapsed: {:.2f}s".format(time.time() - start_time))
