# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Load the T5 tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

# # You would first tokenize your input and output texts (recipe details and tags)
# train_texts = ["Ingredients and instructions..."]  # Example text input
# train_tags = ["soup, dinner, comfort food"]  # Example tags output

# # Tokenize the inputs and outputs
# inputs = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True)
# labels = tokenizer(train_tags, return_tensors="pt", padding=True, truncation=True).input_ids

# # Fine-tune T5 model (this step is resource-intensive)
# outputs = model(input_ids=inputs.input_ids, labels=labels)

