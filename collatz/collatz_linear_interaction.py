import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
n_vals = np.arange(1, N+1)
parity = n_vals % 2
# Features: n, parity, n*parity (interaction)
X = np.column_stack([n_vals, parity, n_vals * parity])
y = np.array([collatz_next(n) for n in n_vals])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Linear Regression
print("Training LinearRegression with interaction...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
# Since we expect exact integers, check accuracy after rounding
y_pred_rounded = np.round(y_pred).astype(int)
acc = accuracy_score(y_test, y_pred_rounded)
print(f"Mean Absolute Error: {mae:.10f}")
print(f"Accuracy (exact match after rounding): {acc*100:.6f}%")
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")

# Function to generate full Collatz sequence using the model
def collatz_sequence_model(start_n, max_steps=1000):
    sequence = [start_n]
    current = start_n
    for _ in range(max_steps):
        parity_cur = current % 2
        X_current = np.array([[current, parity_cur, current * parity_cur]])
        next_pred = model.predict(X_current)[0]
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
model_path = os.path.join(os.path.dirname(__file__), 'collatz_linear_interaction_model.joblib')
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Example usage
print("\nExample: Collatz sequence for 27 using model:")
seq_27 = collatz_sequence_model(27)
print(f"Sequence length: {len(seq_27)}")
print(f"First 10: {seq_27[:10]}")
print(f"Last 10: {seq_27[-10:]}")

# Verify it matches true sequence
seq_true_27 = []
curr = 27
while curr != 1:
    seq_true_27.append(curr)
    curr = collatz_next(curr)
seq_true_27.append(1)
print(f"\nTrue sequence length: {len(seq_true_27)}")
print(f"Sequences match: {seq_27 == seq_true_27}")