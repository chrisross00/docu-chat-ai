from flask import Flask, request, render_template
from transformers import BertForQuestionAnswering, BertTokenizer, RobertaForQuestionAnswering, RobertaTokenizer, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import json

app = Flask(__name__)

# Load the configuration file
with open('config.json') as f:
    config = json.load(f)

# Get the model name from the config
model_name = config['model_name']

# Check the model type and load the appropriate model and tokenizer
if model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
elif model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Specify the path to your saved model
model_path = "../model.pth"

# Load the state dict of your saved model
model.load_state_dict(torch.load(model_path))

# Put the model in evaluation mode
model.eval()

# Create the pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the question and context from the POST request
        question = request.form['question']
        context = request.form['context']

        # Generate the prediction
        output = nlp({
            'question': question,
            'context': context
        })

        # Return the prediction
        return {
            'answer': output['answer'],
            'score': output['score'],
            'start': output['start'],
            'end': output['end']
        }

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
