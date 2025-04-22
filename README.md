# Handwriting Recognition AI

An end-to-end Python application that trains a convolutional neural network on the EMNIST dataset to recognize handwritten digits and letters, and provides a simple Tkinter GUI for live inference.

---

## Features

- **CNN Model** trained on EMNIST Balanced (47 classes: 0–9, A–Z, a–z)  
- **Data Augmentation**: Random rotations, translations, and zooms for better generalization  
- **TensorBoard** integration for visualizing loss, accuracy, histograms, and model graph  
- **Tkinter GUI**: 280×280 drawing canvas with real-time prediction and confidence scoring  

## Images
![image of program in work](https://cs1033.gaul.csd.uwo.ca/~wgao226/images/Screenshot%202025-04-22%20065354.png)
![image of tensorboard](https://cs1033.gaul.csd.uwo.ca/~wgao226/images/Screenshot%202025-04-22%20073905.png)
The plot shows both training (grey) and validation (teal) accuracy over 10 epochs. Training accuracy climbs steadily from about 82% at epoch 0 to roughly 91% by epoch 9, indicating the model is successfully learning the EMNIST patterns. Validation accuracy also improves—from ~82% up to ~87%—but at a slightly slower rate, which is expected since the model hasn’t seen those examples during training. The persistent ~4% gap between training and validation suggests a modest degree of overfitting; beyond epoch 6 both curves begin to flatten, signaling that additional epochs yield diminishing returns. Overall, the model achieves ~87% validation accuracy in under 6 minutes, demonstrating a good balance between learning capacity and generalization on unseen data.
