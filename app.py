from flask import Flask, request
from transformers import BertForQuestionAnswering, BertTokenizer

app = Flask(__name__)

# Specify the path to your saved model
model_path = "/model.pth"

# Specify the path to your BERT base model
bert_base_model_path = 'bert-base-uncased'

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_base_model_path)

# Load the model
model = BertForQuestionAnswering.from_pretrained(bert_base_model_path)
model.load_state_dict(torch.load(model_path))

# Create the pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the question and context from the POST request
    question = request.json['question']
    context = request.json['context']

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

if __name__ == '__main__':
    app.run(debug=True)
