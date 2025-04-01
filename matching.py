import cv2
import numpy as np
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Load trained fingerprint model
MODEL_PATH = "models/fingerprint_model.h5"
model = load_model(MODEL_PATH)

# ✅ Convert BMP to PNG if needed
def convert_to_png(img_path):
    """Converts BMP to PNG for compatibility"""
    if img_path.lower().endswith('.bmp'):
        png_path = img_path.replace('.bmp', '.png')
        img = Image.open(img_path)
        img.save(png_path, "PNG")
        print(f"✅ Converted {img_path} to {png_path}")
        return png_path
    return img_path


# ✅ Validate and load images
def validate_and_load(img_path):
    """Validates image path and loads it"""
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None

    # Convert BMP to PNG if needed
    img_path = convert_to_png(img_path)

    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return None

    return img


# ✅ Preprocessing function
def preprocess_image(img_path, img_size=(128, 128)):
    """Prepares image for fingerprint model"""
    img = validate_and_load(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, img_size)  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img


# ✅ Fingerprint matching function using trained model
def match_fingerprints(img1_path, img2_path):
    """Compares two fingerprint images using a trained model"""

    # Preprocess images
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    if img1 is None or img2 is None:
        print("❌ One or both images could not be processed.")
        return

    # ✅ Get feature embeddings from the model
    feature1 = model.predict(img1)[0]
    feature2 = model.predict(img2)[0]

    # ✅ Calculate similarity score (Cosine Similarity)
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

    # ✅ Display results
    match_score = similarity * 100
    print(f"\n✅ Match Confidence Score: {match_score:.2f}%")

    if match_score > 70:
        print("✅ ✅ Fingerprints Match! ✅ ✅")
    else:
        print("❌ Fingerprints Do NOT Match.")


# ✅ Main Execution
if __name__ == "__main__":
    # 👉 Enter paths for fingerprint images
    img1_path = "data/train2/1/Fingerprint1/1__M_Left_index_finger.BMP"
    img2_path = "data/train2/1/Fingerprint1/1__M_Left_index_finger.BMP"

    # ✅ Match fingerprints using trained model
    match_fingerprints(img1_path, img2_path)
