#!/usr/bin/env python3
"""
main.py

Handwriting Recognition AI:
- Train a CNN on EMNIST (digits + letters)
- Launch a Tkinter drawing canvas for live inference
- Integrate TensorBoard for training visualization
"""
import argparse
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw
import tkinter as tk
TensorBoard = tf.keras.callbacks.TensorBoard
# --- Constants ---
IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 10
MODEL_PATH = 'handwriting_model.h5'

# Global info for label mapping
global emnist_info
emnist_info = None

# --- Data Loading & Preprocessing ---
def load_data():
    global emnist_info
    # Load EMNIST Balanced: 47 classes (0-9, A-Z, a-z)
    (train_ds, test_ds), emnist_info = tfds.load(
        'emnist/balanced',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    def preprocess(image, label):
        # Normalize and reshape
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        # EMNIST images are transposed; rotate to upright
        image = tf.image.rot90(image, k=1)
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        return tf.reshape(image, [IMG_SIZE, IMG_SIZE, 1]), label

    train_ds = train_ds.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
    test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)
    return train_ds, test_ds

# --- Model Definition ---
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(emnist_info.features['label'].num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Training ---
def train(model, train_ds, test_ds):
    log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_cb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[tensorboard_cb]
    )
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# --- Inference Helpers ---
def label_to_char(label):
    names = emnist_info.features['label'].names
    return names[label]

def predict_image(model, pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE)).convert('L')
    arr = np.array(img) / 255.0
    arr = np.rot90(arr, k=-1)  # rotate back
    arr = np.expand_dims(arr, axis=(0, -1))
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return idx, preds[0][idx]

# --- GUI Application ---
def run_app(model):
    root = tk.Tk()
    root.title('Handwriting Recognition')

    canvas = tk.Canvas(root, width=280, height=280, bg='white')
    canvas.pack()

    image = Image.new('RGB', (280, 280), 'white')
    draw = ImageDraw.Draw(image)

    def paint(event):
        x, y = event.x, event.y
        r = 8
        canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        draw.ellipse((x-r, y-r, x+r, y+r), fill='black')

    def clear_canvas():
        canvas.delete('all')
        draw.rectangle((0, 0, 280, 280), fill='white')

    result_var = tk.StringVar()
    result_var.set('Draw a letter or digit and click Predict')
    tk.Label(root, textvariable=result_var, font=('Arial', 14)).pack()

    def recognize():
        idx, prob = predict_image(model, image)
        char = label_to_char(idx)
        result_var.set(f'Prediction: {char} ({prob*100:.2f}%)')

    btn_frame = tk.Frame(root)
    tk.Button(btn_frame, text='Predict', command=recognize).pack(side='left', padx=5)
    tk.Button(btn_frame, text='Clear', command=clear_canvas).pack(side='left', padx=5)
    btn_frame.pack(pady=10)

    canvas.bind('<B1-Motion>', paint)
    root.mainloop()

# --- Entry Point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the CNN')
    args = parser.parse_args()

    train_ds, test_ds = load_data()
    model = build_model()

    if args.train:
        train(model, train_ds, test_ds)
    else:
        if not os.path.exists(MODEL_PATH):
            print(f"No model found at {MODEL_PATH}, please run with --train first.")
            return
        model = keras.models.load_model(MODEL_PATH)
        run_app(model)

if __name__ == '__main__':
    main()
