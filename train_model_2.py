import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# ✅ Function to check if the image is valid
def is_valid_image(image_path):
    """Checks if the image is valid."""
    try:
        img = Image.open(image_path)
        img.verify()  # Verify if the image is corrupted
        return True
    except Exception:
        return False


# ✅ Load and preprocess BMP images from dataset
def load_dataset(data_dir):
    """Loads and preprocesses fingerprint dataset."""
    images = []
    labels = []

    label_map = {}  # Maps class names to continuous indices
    current_label = 0

    # Iterate over folders
    for class_folder in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_folder)

        if os.path.isdir(class_path):
            if class_folder not in label_map:
                label_map[class_folder] = current_label
                current_label += 1

            # Iterate over subfolders (fingerprint1, fingerprint2, ...)
            for fingerprint_folder in os.listdir(class_path):
                fingerprint_path = os.path.join(class_path, fingerprint_folder)

                if os.path.isdir(fingerprint_path):
                    for img_name in os.listdir(fingerprint_path):
                        img_path = os.path.join(fingerprint_path, img_name)

                        # Load BMP images only
                        if img_path.lower().endswith('.bmp') and is_valid_image(img_path):
                            try:
                                # Load and preprocess the image
                                img = Image.open(img_path).convert('L')  # Grayscale
                                img = img.resize((128, 128))  # Resize to 128x128
                                img_array = np.array(img) / 255.0  # Normalize (0-1)

                                images.append(img_array)
                                labels.append(label_map[class_folder])

                            except Exception as e:
                                print(f"❌ Error processing {img_path}: {e}")

    images = np.array(images).reshape(-1, 128, 128, 1)
    labels = np.array(labels)

    print(f"✅ Loaded {len(images)} images with {len(np.unique(labels))} classes.")
    return images, labels


# ✅ CNN Model
def build_model(input_shape, num_classes):
    """Builds a CNN model."""
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer with dynamic classes
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ✅ Train the model
def train():
    """Main training process."""
    data_dir = "data/train3"  # Base directory for dataset

    # Load the dataset
    images, labels = load_dataset(data_dir)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Dynamically set the number of classes
    num_classes = len(np.unique(labels))
    print(f"🔥 Training model with {num_classes} classes.")

    # Build the CNN model
    input_shape = (128, 128, 1)
    model = build_model(input_shape, num_classes)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )

    # ✅ Save the model
    model.save("fingerprint_model2.h5")
    print("✅ Model saved successfully!")

    # ✅ Plot training history
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ✅ Run the training
if __name__ == "__main__":
    train()
