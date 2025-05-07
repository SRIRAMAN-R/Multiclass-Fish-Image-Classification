import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix


def load_data(image_size=(224, 224), batch_size=32):
    """Load training, validation, and test datasets."""
    data_path = r'D:\Mainboot Project\project 5\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data'

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        directory=os.path.join(data_path, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_test_gen.flow_from_directory(
        directory=os.path.join(data_path, 'val'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = val_test_gen.flow_from_directory(
        directory=os.path.join(data_path, 'test'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data


def build_cnn_model(input_shape, num_classes):
    """Build and return a simple CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_transfer_model(base_model_fn, input_shape, num_classes):
    """Build and return a transfer learning model."""
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_and_save_models(train_data, val_data, input_shape, num_classes, cnn_epochs=10, transfer_epochs=5):
    """Train CNN and transfer learning models with different epochs, then save them."""
    os.makedirs("models", exist_ok=True)

    # Train simple CNN
    print("\nTraining Basic CNN...")
    cnn_model = build_cnn_model(input_shape, num_classes)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, validation_data=val_data, epochs=cnn_epochs)
    cnn_model.save("models/cnn_fish_model.h5")

    # Transfer learning models
    models = {
        "vgg16": tf.keras.applications.VGG16,
        "resnet50": tf.keras.applications.ResNet50,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "efficientnetb0": tf.keras.applications.EfficientNetB0
    }

    for name, model_fn in models.items():
        print(f"\nTraining {name.upper()}...")
        model = build_transfer_model(model_fn, input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, validation_data=val_data, epochs=transfer_epochs)
        model.save(f"models/{name}_fish_model.h5")

def evaluate_models(test_data):
    """Evaluate all saved models using classification metrics."""
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    true_labels = test_data.classes
    class_labels = list(test_data.class_indices.keys())

    print("\nModel Evaluation Summary:\n")

    for model_file in model_files:
        print(f"Evaluating {model_file}...")
        model = tf.keras.models.load_model(os.path.join(model_dir, model_file))

        preds = model.predict(test_data)
        predicted_labels = np.argmax(preds, axis=1)

        report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
        matrix = confusion_matrix(true_labels, predicted_labels)

        acc = report['accuracy']
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
        print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
        print(f"F1-score (macro): {report['macro avg']['f1-score']:.4f}")
        print(f"Confusion Matrix:\n{matrix}")
        print("-" * 50)

if __name__ == "__main__":
    IMG_SHAPE = (224, 224, 3)

    train_data, val_data, test_data = load_data()
    num_classes = train_data.num_classes

    train_and_save_models(train_data, val_data, IMG_SHAPE, num_classes, cnn_epochs=10, transfer_epochs=5)
    evaluate_models(test_data)
