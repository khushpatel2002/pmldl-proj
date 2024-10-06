import os
import matplotlib.pyplot as plt
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the trained model
saved_model_path = '/Users/khushpatel2002/pmldl-proj-main/notebooks/models/t5-baseline/checkpoint-3645'
model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
tokenizer = T5TokenizerFast.from_pretrained(saved_model_path)

# Move the model to the appropriate device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)



# Load dataset and split into train and test
full_data = pd.read_csv("hf://datasets/s-nlp/paradetox/train.tsv", sep="\t")
train_data, test_data = train_test_split(full_data, test_size=0.1, random_state=42)  # Using a small part for testing

# Prepare test data
x_test = tokenizer(test_data["en_toxic_comment"].tolist(), return_tensors="pt", truncation=True, padding=True)
y_test = test_data["en_neutral_comment"].tolist()

# Move inputs to device
x_test = {k: v.to(device) for k, v in x_test.items()}




# Make predictions
model.eval()
predictions = []
for i in range(len(x_test["input_ids"])):
    inputs = {k: v[i:i+1] for k, v in x_test.items()}  # Select one example at a time
    output = model.generate(**inputs, num_return_sequences=1, do_sample=False, num_beams=5)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(prediction)




# Evaluate using BLEU and ROUGE
references = [ref.split() for ref in y_test]  # Convert references to list of tokens
predictions_split = [pred.split() for pred in predictions]

# BLEU Score Calculation
bleu_scores = [sentence_bleu([ref], pred) for ref, pred in zip(references, predictions_split)]
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

# ROUGE Score Calculation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(y_test, predictions)]

average_rouge_score = {
    "rouge1": sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores),
    "rouge2": sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores),
    "rougeL": sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
}

print("Average BLEU Score:", average_bleu_score)
print("Average ROUGE Scores:", average_rouge_score)

# Plotting Average Scores
plt.figure(figsize=(10, 6))

# Plotting Average BLEU Score
plt.bar(['BLEU'], [average_bleu_score], color='b', alpha=0.6)

# Plotting Average ROUGE Scores
rouge_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
rouge_values = [average_rouge_score['rouge1'], average_rouge_score['rouge2'], average_rouge_score['rougeL']]
plt.bar(rouge_labels, rouge_values, color=['r', 'g', 'c'], alpha=0.6)

plt.ylabel('Score')
plt.title('Average BLEU and ROUGE Scores')
plt.grid(axis='y')
plt.show()

# Additional Graph: Prediction Lengths vs Target Lengths
pred_lengths = [len(p.split()) for p in predictions]
target_lengths = [len(t.split()) for t in y_test]

plt.figure(figsize=(10, 6))
plt.hist(pred_lengths, alpha=0.5, bins=30, label='Predicted Lengths')
plt.hist(target_lengths, alpha=0.5, bins=30, label='Target Lengths')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.title('Comparison of Prediction and Target Lengths')
plt.legend()
plt.grid(True)
plt.show()