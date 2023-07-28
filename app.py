from flask import Flask, request, render_template
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import torch

app = Flask(__name__)

# Specify the path to your saved model
model_path = "model.pth"

# Specify the path to your BERT base model
bert_base_model_path = 'bert-base-uncased'

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_base_model_path)

# Load the BERT QA Model
model = BertForQuestionAnswering.from_pretrained(bert_base_model_path)

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
