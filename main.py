import os
import pickle
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import BertForQuestionAnswering, BertTokenizer, RobertaForQuestionAnswering, RobertaTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline


# Download necessary NLTK data
nltk.download('punkt')  # Download the tokenizer model
nltk.download('stopwords')  # Download stopwords
nltk.download('wordnet')  # Download WordNet lemmatizer data

def load_squad_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    contexts = []
    questions = []
    answers = []
    for group in data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    print(f'--load_squad_data done!--')
    return contexts, questions, answers

def preprocess_data(contexts, questions):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    preprocessed_contexts = []
    for context in contexts:
        # Tokenize
        word_tokens = word_tokenize(context.lower())
        # Remove punctuation
        word_tokens = [word for word in word_tokens if word.isalnum()]
        # Remove stopwords
        word_tokens = [word for word in word_tokens if not word in stop_words]
        # Perform lemmatization
        word_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
        preprocessed_contexts.append(word_tokens)
        
    preprocessed_questions = []
    for question in questions:
        # Tokenize
        word_tokens = word_tokenize(question.lower())
        # Remove punctuation
        word_tokens = [word for word in word_tokens if word.isalnum()]
        # Remove stopwords
        word_tokens = [word for word in word_tokens if not word in stop_words]
        # Perform lemmatization
        word_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
        preprocessed_questions.append(word_tokens)
    
    
    print(f'--preprocess_data done!--')
    return preprocessed_contexts, preprocessed_questions

def encode_data(tokenizer, questions, passages, max_length):
    """Encode the question/passage pairs into features that can be fed into the model."""
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(
            question, 
            passage, 
            max_length=max_length, 
            padding='max_length', 
            truncation='longest_first', 
            return_attention_mask=True
        )
        input_ids.append(encoded_data['input_ids'])
        attention_masks.append(encoded_data['attention_mask'])

    print(f'--encode_data done!--')
    return input_ids, attention_masks


## Execution
## Preprocess, Extract, Build, Train 

# Checkpoint file names
preprocessed_data_file = './preprocessed_data.pkl'

# Check if preprocessed data already exists
if os.path.exists(preprocessed_data_file):
    # Load preprocessed data from file
    with open(preprocessed_data_file, 'rb') as f:
        contexts, questions, answers = load_squad_data('../train-v1.1.json')
        preprocessed_contexts, preprocessed_questions, answers = pickle.load(f)
else:
    # Replace 'train-v1.1.json' with the path to your SQuAD data file
    contexts, questions, answers = load_squad_data('../train-v1.1.json')
    preprocessed_contexts, preprocessed_questions = preprocess_data(contexts, questions)

    # Save preprocessed data to file
    with open(preprocessed_data_file, 'wb') as f:
        pickle.dump((preprocessed_contexts, preprocessed_questions, answers), f)


# Load the configuration file
with open('config.json') as f:
    config = json.load(f)

# Get the model name from the config
model_name = config['model_name']
print(model_name)

# Check the model type and load the appropriate model and tokenizer
if model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
elif model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)
else:
    raise ValueError(f'Unsupported model type for name: {model_name}')

# Encode the data
max_seq_length = 256  # choose a maximum sequence length that fits your memory constraints
input_ids, attention_masks = encode_data(tokenizer, questions, contexts, max_seq_length)

# Calculate start and end positions for each answer
start_positions = [answer['answer_start'] for answer in answers]
end_positions = [start + len(answer['text']) for start, answer in zip(start_positions, answers)]

# Convert to tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)

# Create the DataLoader
dataset = TensorDataset(input_ids, attention_masks, start_positions, end_positions)
train_dataloader = DataLoader(dataset, batch_size=16)  # adjust batch size as needed

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate

# Define the number of training epochs
epochs = 3

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Number of training epochs
epochs = 3

# Store the average loss after each epoch so we can plot them
loss_values = []
print_every = 100  # adjust this to print loss every n steps

# For each epoch...
for epoch in range(0, epochs):
    print(f'======== Epoch {epoch + 1} / {epochs} ========')

    # Measure the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Clear any previously calculated gradients before backward pass
        optimizer.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch)
        outputs = model(input_ids=batch[0].to('cuda'),
                        attention_mask=batch[1].to('cuda'),
                        start_positions=batch[2].to('cuda'),
                        end_positions=batch[3].to('cuda'))

        # Accumulate the training loss over all of the batches
        total_loss += outputs[0].item()

        # Perform a backward pass to calculate the gradients
        outputs[0].backward()

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Print loss every `print_every` steps
        if step % print_every == 0 and not step == 0:
            # Calculate the average loss over the training data
            avg_train_loss = total_loss / print_every
            print(f'Step {step} - Average training loss: {avg_train_loss}')

            # Reset the total loss for the next set of `print_every` steps
            total_loss = 0

    # Print the average loss over the entire epoch
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss for epoch: {avg_train_loss}')

    # Store the loss value for plotting the learning curve
    loss_values.append(avg_train_loss)
    # Save the model 
    torch.save(model.state_dict(), '../model.pth')

print('Training complete!')