# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-math/MetaMath-7B-V1.0")

# Example prompt
prompt = "Solve the equation: 2x + 3 = 11"

# Generate text using the model
output = pipe(prompt, max_length=50)

# Print the generated text
print(output[0]['generated_text'])