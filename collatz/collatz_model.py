import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os

# Collatz next function
def collatz_next(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

# Generate data
print("Generating dataset...")
N = 10000  # Numbers from 1 to 10000
X = np.array(range(1, N+1)).reshape(-1, 1)
y = np.array([collatz_next(n) for n in range(1, N+1)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
print("Training MLPRegressor...")
model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', 
                     max_iter=2000, random_state=42, verbose=False)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)  # Round to nearest integer

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred_rounded)
print(f"Mean Absolute Error: {mae:.6f}")
print(f"Accuracy (exact match): {acc*100:.2f}%")

# Check if any predictions are off
if mae > 0.5:
    print("Warning: MAE > 0.5 suggests some predictions are not close to integers.")
else:
    print("MAE is low, predictions are close to integers.")

# Function to generate full Collatz sequence using the model
def collatz_sequence_model(start_n, max_steps=1000):
    sequence = [start_n]
    current = start_n
    for _ in range(max_steps):
        # Predict next value
        next_pred = model.predict(np.array([[current]]))[0]
        next_int = int(np.round(next_pred))
        sequence.append(next_int)
        current = next_int
        if current == 1:  # Collatz conjecture ends at 1
            break
    return sequence

# Test with a few numbers
test_numbers = [1, 2, 3, 10, 27, 97]
print("\nTesting Collatz sequence generation:")
for n in test_numbers:
    seq_model = collatz_sequence_model(n)
    # Generate true sequence for comparison
    seq_true = []
    curr = n
    while curr != 1 and len(seq_true) < 1000:
        seq_true.append(curr)
        curr = collatz_next(curr)
    seq_true.append(1)  # Add the final 1
    
    # Compare
    match = seq_model == seq_true
    print(f"Start {n}: Model sequence length {len(seq_model)}, True length {len(seq_true)}")
    print(f"  Match: {match}")
    if not match:
        # Find first difference
        for i, (m, t) in enumerate(zip(seq_model, seq_true)):
            if m != t:
                print(f"  First difference at step {i}: model={m}, true={t}")
                break

# Save model
model_path = os.path.join(os.path.dirname(__file__), 'collatz_model.joblib')
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Example usage
print("\nExample: Collatz sequence for 27 using model:")
seq_27 = collatz_sequence_model(27)
print(f"Sequence length: {len(seq_27)}")
print(f"First 10: {seq_27[:10]}")
print(f"Last 10: {seq_27[-10:]}")