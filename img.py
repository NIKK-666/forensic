import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Load fingerprint model
MODEL_PATH = "models/fingerprint_model.h5"
model = load_model(MODEL_PATH)

# ✅ Convert BMP to PNG safely
def sanitize_filename(filename):
    """Removes invalid characters for Windows filenames"""
    invalid_chars = r'\/:*?"<>|'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename[:200]  # Shorten if too long

def convert_to_png(img_path):
    """Converts BMP to PNG while handling invalid characters"""
    if img_path.lower().endswith('.bmp'):
        folder, filename = os.path.split(img_path)
        filename = sanitize_filename(filename)
        png_path = os.path.join(folder, filename.replace('.bmp', '.png'))

        try:
            img = Image.open(img_path)
            img.save(png_path, "PNG")
            print(f"✅ Converted {img_path} to {png_path}")
            return png_path
        except Exception as e:
            print(f"❌ Failed to convert {img_path}: {e}")
            return None
    return img_path

# ✅ Load and validate images
def validate_and_load(img_path):
    """Validates image path and loads it"""
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None

    # Convert BMP to PNG safely
    img_path = convert_to_png(img_path)

    if img_path is None or not os.path.exists(img_path):
        print(f"❌ Failed to process image: {img_path}")
        return None

    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return None

    return img

# ✅ Minutiae extraction function
def extract_minutiae(image_gray):
    """Extracts minutiae points (ridge endings & bifurcations)"""
    # Apply adaptive thresholding
    img_bin = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Skeletonization
    skeleton = skeletonize(img_bin // 255)

    # Detect local maxima (potential minutiae points)
    minutiae = peak_local_max(skeleton.astype(np.uint8), min_distance=5, threshold_abs=0.1)

    return minutiae, skeleton

# ✅ Preprocessing function
def preprocess_image(img_path, img_size=(128, 128)):
    """Prepares image for fingerprint model and extracts minutiae"""
    img = validate_and_load(img_path)

    if img is None:
        return None, None, None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, img_size)
    img_norm = img_resized / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)
    img_expanded = np.expand_dims(img_expanded, axis=-1)

    # Detect minutiae points
    minutiae, skeleton = extract_minutiae(img_gray)

    return img_expanded, img, minutiae

# ✅ Matching function
def match_fingerprints(img1_path, img2_path):
    """Compares two fingerprint images using a trained model and shows minutiae points"""
    img1, img1_display, minutiae1 = preprocess_image(img1_path)
    img2, img2_display, minutiae2 = preprocess_image(img2_path)

    if img1 is None or img2 is None:
        print("❌ One or both images could not be processed.")
        return

    # ✅ Get feature embeddings from the model
    feature1 = model.predict(img1)[0]
    feature2 = model.predict(img2)[0]

    # ✅ Calculate similarity score
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    match_score = similarity * 100

    # ✅ Display result
    print(f"\n✅ Match Confidence Score: {match_score:.2f}%")

    if match_score > 70:
        print("✅ ✅ Fingerprints Match! ✅ ✅")
    else:
        print("❌ Fingerprints Do NOT Match.")

    # ✅ Show images with minutiae points
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # ✅ Draw minutiae points
    for ax, img, minutiae, title in zip(axs, [img1_display, img2_display], [minutiae1, minutiae2], ["Fingerprint 1", "Fingerprint 2"]):
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.scatter(minutiae[:, 1], minutiae[:, 0], c='red', s=10, label="Minutiae")
            ax.legend()
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Match Score: {match_score:.2f}%", fontsize=14, fontweight="bold", color="blue" if match_score > 70 else "red")
    plt.show()

# ✅ Main Execution
if __name__ == "__main__":
    img1_path = "data/train2/3/Fingerprint3/3__M_Left_index_finger.BMP"
    img2_path = "data/train2/3/Fingerprint3/3__M_Left_index_finger.BMP"
    match_fingerprints(img1_path, img2_path)
