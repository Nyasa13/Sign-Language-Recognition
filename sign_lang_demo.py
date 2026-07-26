import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)


# CONFIGURATION
DATASET_PATH = "dataset/asl_alphabet_train"
IMG_SIZE     = 224     
BATCH_SIZE   = 64      
FINE_TUNE_AT = 125     


CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]
NUM_CLASSES = len(CLASS_NAMES)

def create_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range         = 15,
        zoom_range             = 0.15,
        width_shift_range      = 0.10,
        height_shift_range     = 0.10,
        brightness_range       = [0.80, 1.20],
        fill_mode              = 'nearest',
        validation_split       = 0.15,
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        validation_split       = 0.15,
    )

    print("[INFO] Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        subset      = 'training',
        shuffle     = True,
        seed        = 42,
    )

    print("[INFO] Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        subset      = 'validation',
        shuffle     = False,
        seed        = 42,
    )

    print("\n[INFO] Class index mapping (verify this matches CLASS_NAMES order):")
    for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"  [{idx:2d}] {name}")

    return train_gen, val_gen


# BUILD MODEL
def build_model():
    base_model = MobileNetV2(
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
        include_top = False,
        weights     = 'imagenet',
    )

    base_model.trainable = False
    print(f"[INFO] MobileNetV2 total layers: {len(base_model.layers)}")
    print(f"[INFO] Phase 1: all {len(base_model.layers)} backbone layers frozen")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)     
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.3)(x)                  
    x = Dense(64,  activation='relu')(x) 
    x = Dropout(0.2)(x)                  
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"[INFO] Trainable layers: {trainable} / {len(model.layers)}")
    return model, base_model

#PHASE-1 TRAINING
def phase1_train(model, train_gen, val_gen):
    print("\n" + "="*55)
    print("PHASE 1: Training classification head (backbone frozen)")
    print("="*55)

    callbacks = [
        ModelCheckpoint('best_phase1.keras',
                        monitor='val_accuracy', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=3,
                      restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        epochs          = 5,
        validation_data = val_gen,
        callbacks       = callbacks,
        verbose         = 1,
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\n[Phase 1 done] Best val accuracy: {val_acc*100:.2f}%")
    return history

#PHASE-2 TRAINING
def phase2_train(model, base_model, train_gen, val_gen):
    print("\n" + "="*55)
    print("PHASE 2: Fine-tuning last 30 backbone layers")
    print(f"         Unfreezing from layer {FINE_TUNE_AT} onward")
    print("="*55)

    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False   

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] Unfrozen backbone layers: {unfrozen}")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )

    callbacks = [
        ModelCheckpoint('sign_model_tf',
                        monitor='val_accuracy', save_best_only=True,
                        verbose=1, save_format='tf'),
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                         patience=3, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_gen,
        epochs          = 30,
        validation_data = val_gen,
        callbacks       = callbacks,
        verbose         = 1,
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\n[Phase 2 done] Best val accuracy: {val_acc*100:.2f}%")
    return history

# EVALUATE
def evaluate(model):
    print("\n[INFO] Final evaluation...")
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    eval_gen = eval_datagen.flow_from_directory(
        DATASET_PATH,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        shuffle     = False,
    )
    loss, acc = model.evaluate(eval_gen, verbose=1)
    print(f"\n[RESULT] Accuracy : {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(eval_gen, verbose=0), axis=1)
    y_true = eval_gen.classes
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# PLOT
def plot(h1, h2):
    acc  = h1.history['accuracy']     + h2.history['accuracy']
    vacc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss']         + h2.history['loss']
    vloss= h1.history['val_loss']     + h2.history['val_loss']
    split = len(h1.history['accuracy'])  # mark phase boundary

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train, val, title in [
        (axes[0], acc,  vacc, 'Accuracy'),
        (axes[1], loss, vloss,'Loss'),
    ]:
        ax.plot(train, label='Train', color='steelblue',  linewidth=2)
        ax.plot(val,   label='Val',   color='darkorange', linewidth=2)
        ax.axvline(split, color='gray', linestyle='--', label='Fine-tune start')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=120)
    print("[INFO] Saved training_curves.png")
    plt.show()

# MAIN
if __name__ == "__main__":

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        print("  1. Download from kaggle.com/datasets/grassknoted/asl-alphabet")
        print("  2. Extract into:  dataset/asl_alphabet_train/")
        exit(1)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("[INFO] No GPU — CPU only (Phase 1: ~20 min, Phase 2: ~60 min)")
        print("       On GPU: Phase 1 ~3 min, Phase 2 ~15 min")

    train_gen, val_gen    = create_generators()
    model, base_model     = build_model()
    h1                    = phase1_train(model, train_gen, val_gen)
    h2                    = phase2_train(model, base_model, train_gen, val_gen)

    evaluate(model)
    plot(h1, h2)

    model.save("sign_model_tf")
    print("\n[DONE] Model saved to sign_model_tf/")
    print("       Run: python realtime_sign.py")