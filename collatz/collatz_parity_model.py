import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Collatz next function
def collatz_next(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

# Generate data for parity prediction
print("Generating dataset for parity...")
N = 10000  # Numbers from 1 to 10000
X = np.array(range(1, N+1)).reshape(-1, 1)
# Target: 0 for even, 1 for odd
y = np.array([n % 2 for n in range(1, N+1)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train logistic regression model
print("Training LogisticRegression...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Function to generate full Collatz sequence using the model
def collatz_sequence_model(start_n, max_steps=1000):
    sequence = [start_n]
    current = start_n
    for _ in range(max_steps):
        # Predict parity (0 even, 1 odd)
        parity_pred = model.predict(np.array([[current]]))[0]
        parity_pred = int(round(parity_pred))  # ensure 0 or 1
        if parity_pred == 0:
            next_val = current // 2
        else:
            next_val = 3 * current + 1
        sequence.append(next_val)
        current = next_val
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
model_path = os.path.join(os.path.dirname(__file__), 'collatz_parity_model.joblib')
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Example usage
print("\nExample: Collatz sequence for 27 using model:")
seq_27 = collatz_sequence_model(27)
print(f"Sequence length: {len(seq_27)}")
print(f"First 10: {seq_27[:10]}")
print(f"Last 10: {seq_27[-10:]}")