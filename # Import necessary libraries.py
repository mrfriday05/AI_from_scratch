import tensorflow as tf
import numpy as np
import math

NUM_SAMPLES=1000
TEST_FEATURES=4

x_train = np.sqrt(np.random.rand(NUM_SAMPLES, 3).astype(np.float32))
x_test = np.sqrt(np.random.rand(TEST_FEATURES, 3).astype(np.float32))

y_train = np.zeros((NUM_SAMPLES, 3), dtype=np.float32) 

for i in range(NUM_SAMPLES):
    
    current_input = x_train[i]

    max_val = np.max(current_input)

    min_val = np.min(current_input)

    avg_val = np.mean(current_input)

    y_train[i] = [max_val, min_val, avg_val]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='relu'), 
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

print("--- Model Summary ---")
model.summary()
print("\n")

print("--- Starting Training ---")
history=model.fit(x_train, y_train, epochs=1000, verbose=0)
print("--- Training Finished ---")

final_loss = history.history['loss'][-1]
final_accuracy = history.history['mse'][-1]
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Accuracy: {final_accuracy:.4f}\n")


print("--- Model Predictions ---")
predictions_raw = model.predict(x_test)
for i in range(len(x_test)):
    input_data = x_test[i]
    raw_prediction = predictions_raw[i]
    print (f"input data:{input_data}")
    print(f"Output data: {raw_prediction}")
    print(f"Correct data: nem kapod meg")


model_save_path = "models\minmax.keras"

# Save the entire model to the specified path.
print(f"--- Saving model to directory: {model_save_path} ---")
model.save(model_save_path)
print("--- Model Saved Successfully ---\n")