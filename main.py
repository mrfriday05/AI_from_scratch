import numpy as np
from neural_network.network import Network
import matplotlib.pyplot as plt
from neural_network.layer import Layer
import image_processing.image_import as imp
from pathlib import Path
import random
import time

seed = random.randint(1000, 9999)

H = 28
W = 28
# training_length

filepath = Path("data")
x_train, y_train = imp.import_images(2000, 16, filepath, H, W)

x_test, y_test = imp.import_images(10, 1, filepath, H, W)

print(f"\nTraining data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

load=True
open_name="model1_6090"
if load:
    myNetwork=Network()
    myNetwork.load_network(open_name)
else:    
    myNetwork = Network([
        {"neurons": H * W},
        {"neurons": 128, "activation": "Sigmoid"},
        {"neurons": 64, "activation": "Sigmoid"},
        {"neurons": 32, "activation": "Sigmoid"},
        {"neurons": 10, "activation": "Softmax"}
    ])

print("\nNetwork Architecture:")
print(myNetwork)

# --- 3. Train the Network ---
save=True
save_name="model1"
epochs = 1000
learning_rate = 0.15
lr=learning_rate
losses = []
start=time.time()

print("\nStarting training...")
for i in range(epochs):
    err = myNetwork.learn(x_train, y_train, lr)
    lr = err*learning_rate*0.8+learning_rate*0.2
    losses.append(err)
    if (i + 1) % 100 == 0:
        # Calculate accuracy on test set
        correct_predictions = 0
        for j in range(len(x_test)):
            prediction = myNetwork.compute(x_test[j])
            if np.argmax(prediction) == np.argmax(y_test[j]):
                correct_predictions += 1
        accuracy = (correct_predictions / len(x_test)) * 100
        length=time.time()-start
        print(f"Epoch: {i+1}/{epochs}, Loss: {err:.4f}, Test Accuracy: {accuracy:.2f}%, Time: {length:.0f}s")
    
    if (i+1) % 200 == 0:
        myNetwork.save_network(filename=f"{save_name}_{seed}")

print("Training complete.")
if save:
    myNetwork.save_network(filename=f"{save_name}_{seed}")
    
# --- 4. Visualize Loss ---
'''
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()
'''

# --- 5. Visualize 10 Random Predictions ---
num_images_to_show = 10
# Get random indices from the test set
random_indices = random.sample(range(len(x_test)), num_images_to_show)

# Create a subplot for the images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Random Predictions', fontsize=16)

for i, ax in enumerate(axes.ravel()):
    # Get the random index
    index = random_indices[i]
    
    # Reshape the flattened image back to 28x28 for display
    image = x_test[index].reshape(H, W)
    
    # Get the true label for this image
    true_label = np.argmax(y_test[index])
    
    # Make a prediction with the trained network
    prediction = myNetwork.compute(x_test[index])
    predicted_label = np.argmax(prediction)
    
    # Display the image
    ax.imshow(image, cmap='gray')
    
    # Set the title with the prediction and true label
    title_color = 'green' if predicted_label == true_label else 'red'
    ax.set_title(f"Pred: {predicted_label} | True: {true_label}", color=title_color)
    
    # Hide the axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.plot(losses)
plt.show()