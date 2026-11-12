
from modules.evaluator import build_evaluation_dataset


# Step 1: Build the evaluation dataset
dataset = build_evaluation_dataset()

# Step 2: Display total evaluation samples
print("\nTOTAL EVALUATION SAMPLES:")
print(len(dataset))

# Step 3: Display dataset columns
print("\nDATASET COLUMNS:")
print(dataset.features())

# Step 4: Display the first evaluation sample
print("\nFIRST EVALUATION SAMPLE:")
print(dataset[0])
