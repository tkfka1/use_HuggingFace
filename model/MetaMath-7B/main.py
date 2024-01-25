from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-math/MetaMath-7B-V1.0")
model = AutoModelForCausalLM.from_pretrained("meta-math/MetaMath-7B-V1.0")

# Prepare the prompt
prompt = "Solve the equation: 2x + 3 = 11"

# Encode the prompt text to tensor of integers using the tokenizer
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate a response from the model
output = model.generate(input_ids, max_length=50)

# Decode the generated ids to a string
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)