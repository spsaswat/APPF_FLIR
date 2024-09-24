import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV3Large
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import cv2
from tqdm import tqdm  # Import tqdm for progress bar
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# 1. Define the Segmentation Models
def conv_block(inputs, filters):
    conv = layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
    conv = layers.Conv2D(filters, 3, activation='relu', padding='same')(conv)
    return conv

def unet(input_size=(256, 256, 3)):
    inputs = tf.keras.Input(input_size)
    
    # Encoder (Downsampling)
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = conv_block(pool4, 1024)
    
    # Decoder (Upsampling)
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = conv_block(up6, 512)
    
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = conv_block(up7, 256)
    
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = conv_block(up8, 128)
    
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = conv_block(up9, 64)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    return models.Model(inputs=inputs, outputs=outputs)

def resnet50_unet(input_size=(256, 256, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)
    
    layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False
    
    inputs = tf.keras.Input(input_size)
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    for up, skip in zip([512, 256, 128, 64], skips):
        x = layers.Conv2DTranspose(up, 3, strides=2, padding='same')(x)
        x = layers.concatenate([x, skip])
        x = conv_block(x, up)
    
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def mobilenetv3_unet(input_size=(256, 256, 3)):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_size)
    
    layer_names = ['expanded_conv/output', 'expanded_conv_4/output', 'expanded_conv_8/output', 'expanded_conv_12/output']
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False
    
    inputs = tf.keras.Input(input_size)
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    for up, skip in zip([256, 128, 64], skips):
        x = layers.Conv2DTranspose(up, 3, strides=2, padding='same')(x)
        x = layers.concatenate([x, skip])
        x = conv_block(x, up)
    
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def sam_unet(input_shape, sam_embedding_size):
    inputs = layers.Input(shape=input_shape)
    sam_input = layers.Input(shape=(sam_embedding_size,))
    
    sam_expanded = layers.Dense(input_shape[0] * input_shape[1])(sam_input)
    sam_reshaped = layers.Reshape((input_shape[0], input_shape[1], 1))(sam_expanded)
    x = layers.Concatenate()([inputs, sam_reshaped])
    
    conv1 = conv_block(x, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_block(pool4, 1024)
    
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = conv_block(up6, 512)
    
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = conv_block(up7, 256)
    
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = conv_block(up8, 128)
    
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = conv_block(up9, 64)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    return models.Model(inputs=[inputs, sam_input], outputs=outputs)

# 2. Data Augmentation for Training
def get_training_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

# 3. Test-Time Augmentation (TTA)
def perform_tta(model, image, num_augments=5):
    tta_predictions = []
    for _ in range(num_augments):
        augmented_image = tf.expand_dims(image, 0)
        augmented_image = get_training_augmentation()(augmented_image)
        prediction = model.predict(augmented_image, verbose=0)[0]
        tta_predictions.append(prediction)
    return np.mean(tta_predictions, axis=0)

# 4. Model Certainty Analysis
def calculate_uncertainty(model, image, num_augments=5):
    tta_predictions = []
    for _ in range(num_augments):
        augmented_image = tf.expand_dims(image, 0)
        augmented_image = get_training_augmentation()(augmented_image)
        prediction = model.predict(augmented_image, verbose=0)[0]
        tta_predictions.append(prediction)
    return np.mean(tta_predictions, axis=0), np.var(tta_predictions, axis=0)

# 5. Model Selection and Compilation
def get_model(model_name, input_size=(256, 256, 3), sam_embedding_size=256):
    if model_name == 'unet':
        return unet(input_size)
    elif model_name == 'resnet50_unet':
        return resnet50_unet(input_size)
    elif model_name == 'mobilenetv3_unet':
        return mobilenetv3_unet(input_size)
    elif model_name == 'sam_unet':
        return sam_unet(input_size, sam_embedding_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# 6. Performance Metrics
def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    # Binarize predictions for metrics calculation
    y_pred_flat = np.where(y_pred_flat > 0.5, 1, 0)
    
    iou = jaccard_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    
    return iou, f1

# Function to generate SAM embeddings for each image
def generate_sam_embeddings(sam_predictor, images):
    sam_embeddings = []
    for image in images:
        sam_predictor.set_image(image)
        embedding = sam_predictor.get_image_embedding().cpu().numpy()
        sam_embeddings.append(embedding)
    return np.array(sam_embeddings)

# 7. Training and Evaluation Function
# 7. Training and Evaluation Function (Updated to Use SAM Embeddings)
def train_and_evaluate_model(model_name, X_train, X_val, y_train, y_val, sam_predictor=None, batch_size=16, epochs=50):
    input_size = X_train.shape[1:]

    # Generate SAM embeddings if the model is `sam_unet`
    if model_name == 'sam_unet' and sam_predictor is not None:
        sam_train_embeddings = generate_sam_embeddings(sam_predictor, X_train)
        sam_val_embeddings = generate_sam_embeddings(sam_predictor, X_val)
    else:
        sam_train_embeddings = None
        sam_val_embeddings = None
    
    model = get_model(model_name, input_size=input_size)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Data augmentation
    train_augmentation = get_training_augmentation()

    # Callback for saving the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}_best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    
    # Wrap the training process with tqdm progress bar
    with tqdm(total=epochs, desc=f"Training {model_name}", unit="epoch") as pbar:
        history = model.fit(
            [train_augmentation(X_train), sam_train_embeddings] if model_name == 'sam_unet' else train_augmentation(X_train),
            y_train,
            validation_data=([X_val, sam_val_embeddings], y_val) if model_name == 'sam_unet' else (X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint],
            verbose=0  # Set verbose to 0 to avoid clutter, tqdm will handle the progress
        )
        for epoch in range(epochs):
            pbar.update(1)
    
    # Load best model for evaluation
    model.load_weights(f'{model_name}_best_model.h5')
    
    # Evaluate on validation data
    val_preds = model.predict([X_val, sam_val_embeddings] if model_name == 'sam_unet' else X_val)
    
    iou, f1 = calculate_metrics(y_val, val_preds)
    
    # Test-Time Augmentation (TTA)
    val_preds_tta = np.array([perform_tta(model, img) for img in X_val])
    iou_tta, f1_tta = calculate_metrics(y_val, val_preds_tta)
    
    return {
        'model_name': model_name,
        'history': history.history,
        'iou': iou,
        'f1': f1,
        'iou_tta': iou_tta,
        'f1_tta': f1_tta
    }

# 8. Data Loading and Preparation for Folders of Images and Masks
def load_data_from_folders(image_folder, mask_folder, test_size=0.2):
    images = []
    masks = []

    # List all files in the folder
    all_files = os.listdir(image_folder)

    # Separate image and JSON files
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    json_files = [f for f in all_files if f.lower().endswith('.json')]

    for image_filename in image_files:
        image_path = os.path.join(image_folder, image_filename)
        
        # Find corresponding JSON file
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        if json_filename not in json_files:
            print(f"No corresponding JSON file found for {image_filename}")
            continue

        json_path = os.path.join(mask_folder, json_filename)

        # Load the image
        try:
            image = img_to_array(load_img(image_path, target_size=(256, 256))) / 255.0
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_filename}: {e}")
            continue

        # Load and process the mask from the JSON file
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                mask_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error in file {json_filename}: {e}")
            continue
        except Exception as e:
            print(f"Error reading file {json_filename}: {e}")
            continue

        mask = np.zeros((256, 256), dtype=np.uint8)

        # Extracting points from JSON and creating the mask
        if 'shapes' in mask_data:
            for shape in mask_data['shapes']:
                if 'points' in shape:
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)
                else:
                    print(f"No 'points' key in shape in file {json_filename}")
        else:
            print(f"No 'shapes' key in JSON data in file {json_filename}")
            continue

        mask = np.expand_dims(mask, axis=-1)
        mask = cv2.resize(mask, (256, 256)) / 255.0
        masks.append(mask)

    # Convert lists to NumPy arrays
    images = np.array(images)
    masks = np.array(masks)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=test_size)

    return X_train, X_val, y_train, y_val


# 9. Main Training Loop for Different Models (Updated for SAM)
def main():
    # Initialize the SAM model using sam_model_registry
    sam_checkpoint = "C:/Users/polis/Downloads/Azure_TL/sam_vit_h_4b8939.pth"  # Add the correct path to the SAM checkpoint
    sam_model_type = "vit_h"  # Change this to the correct model type (e.g., "vit_b", "vit_l", etc.)
    
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam_model)

    image_folder = "SensorCommunication/Acquisition/opensource_datasets"
    mask_folder = image_folder
    
    X_train, X_val, y_train, y_val = load_data_from_folders(image_folder, mask_folder)
    
    models_to_train = ['unet', 'resnet50_unet', 'mobilenetv3_unet', 'sam_unet']
    
    results = []
    
    for model_name in models_to_train:
        print(f"Training {model_name}...")
        result = train_and_evaluate_model(model_name, X_train, X_val, y_train, y_val, sam_predictor=sam_predictor if model_name == 'sam_unet' else None)
        results.append(result)
    
        print(f"IoU: {result['iou']:.4f}, F1: {result['f1']:.4f}")
        print(f"IoU (TTA): {result['iou_tta']:.4f}, F1 (TTA): {result['f1_tta']:.4f}")
        print("---")
    
    # Compare the results
    for result in results:
        plt.plot(result['history']['val_accuracy'], label=f"{result['model_name']} - Validation Accuracy")
    
    plt.title("Model Comparison - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Display final metrics for all models
    print("Final Results:")
    for result in results:
        print(f"{result['model_name']}:")
        print(f"  IoU: {result['iou']:.4f}, F1: {result['f1']:.4f}")
        print(f"  IoU (TTA): {result['iou_tta']:.4f}, F1 (TTA): {result['f1_tta']:.4f}")

if __name__ == "__main__":
    main()
