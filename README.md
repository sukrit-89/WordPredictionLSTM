# Word Prediction LSTM RNN

A practical implementation of Long Short-Term Memory (LSTM) Recurrent Neural Network for next-word prediction, trained on Shakespeare's "Hamlet" and deployed with a Streamlit web interface.

## 📋 Project Overview

This project demonstrates the practical application of LSTM neural networks for sequence prediction tasks. The model learns patterns from Shakespeare's "Hamlet" text and predicts the next word in a given sequence of words, making it capable of understanding linguistic patterns and context.

### Key Features

- **Deep LSTM Architecture**: Two-layer LSTM network with dropout regularization
- **Smart Training**: Early stopping and learning rate reduction to prevent overfitting
- **Real-time Predictions**: Interactive web interface powered by Streamlit
- **Optimized Performance**: Embedding layer and batch processing for efficient training
- **Classic Literature Dataset**: Trained on Shakespeare's "Hamlet" (~4,818 unique words)

## 🧠 Model Architecture

The neural network consists of the following layers:

```
Input Layer (13 timesteps)
    ↓
Embedding Layer (4818 → 128 dimensions)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (30%)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dropout (20%)
    ↓
Dense Layer (64 units, ReLU activation)
    ↓
Dropout (20%)
    ↓
Output Layer (4818 units, Softmax activation)
```

**Total Parameters**: 1,115,026 (4.25 MB)

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 16
- **Max Epochs**: 200 (with early stopping)
- **Early Stopping**: Patience of 25 epochs (monitoring validation accuracy)
- **Learning Rate Reduction**: Factor of 0.5 with patience of 10 epochs
- **Train/Test Split**: 80/20

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sukrit-89/WordPredictionLSTM.git
   cd WordPredictionLSTM
   ```

2. **Install required dependencies**
   ```bash
   pip install numpy pandas tensorflow keras nltk streamlit matplotlib
   ```

3. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## 📊 Dataset

The model is trained on **Shakespeare's "Hamlet"**, which provides:
- Rich, complex vocabulary
- Varied sentence structures
- Classic English literature patterns
- Approximately 4,818 unique words

The text is automatically downloaded from NLTK's Gutenberg corpus during the training process.

## 🎯 Usage

### Training the Model

To train the model from scratch:

1. **Run the Jupyter notebook**
   ```bash
   jupyter notebook experiments.ipynb
   ```

2. **Execute all cells** to:
   - Download and preprocess the data
   - Build the LSTM model
   - Train with early stopping
   - Save the trained model and tokenizer

The training process will create:
- `word_prediction_lstm.h5` - Trained model weights
- `tokenizer.pickle` - Fitted tokenizer for text preprocessing

### Running the Web Application

Launch the Streamlit interface for real-time predictions:

```bash
streamlit run app.py
```

This will open a web browser with an interactive interface where you can:
- Enter a sequence of words
- Get instant next-word predictions
- Experiment with different input phrases

### Making Predictions Programmatically

```python
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('word_prediction_lstm.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict next word
def predict_next_word(text):
    max_sequence_len = model.input_shape[1] + 1
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    
    token_list = pad_sequences([token_list], 
                              maxlen=max_sequence_len-1, 
                              padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Example usage
input_text = "To be or not to"
next_word = predict_next_word(input_text)
print(f"Input: {input_text}")
print(f"Predicted next word: {next_word}")
```

## 📈 Model Performance

The model was trained for 36 epochs with the following results:

- **Best Validation Accuracy**: 7.38% (Epoch 11)
- **Final Training Accuracy**: 14.02%
- **Vocabulary Size**: 4,818 words
- **Training Sequences**: 25,732

### Training Insights

The model uses:
- **N-gram sequence generation**: Creates multiple training examples from each sentence
- **Pre-padding**: Ensures consistent input sequence length
- **Dropout layers**: Prevents overfitting by randomly dropping neurons during training
- **Learning rate scheduling**: Automatically reduces learning rate when validation loss plateaus

## 🎨 Example Predictions

| Input Sequence | Predicted Next Word |
|---|---|
| "To be or not to" | "be" |
| "Friends to" | "the" |

*Note: Predictions may vary and depend on the context learned from Shakespeare's writing style*

## 📁 Project Structure

```
WordPredictionLSTM/
│
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Training notebook with full pipeline
├── hamlet.txt                      # Shakespeare's Hamlet text
├── word_prediction_lstm.h5         # Trained model (13.4 MB)
├── tokenizer.pickle                # Fitted tokenizer (187 KB)
└── README.md                       # Project documentation
```

## 🛠️ Technical Details

### Data Preprocessing

1. **Text Tokenization**: Converts words to integer sequences
2. **N-gram Generation**: Creates training sequences of varying lengths
3. **Padding**: Aligns sequences to maximum length (14 words)
4. **One-hot Encoding**: Converts target words to categorical format

### Model Training Steps

1. Load and preprocess Shakespeare's Hamlet
2. Create tokenizer with vocabulary of 4,818 words
3. Generate n-gram sequences for training
4. Pad sequences to uniform length
5. Split into training (80%) and validation (20%) sets
6. Train LSTM model with callbacks (early stopping, learning rate reduction)
7. Save model and tokenizer for deployment

## 🎯 Future Improvements

- [ ] Increase dataset size with more Shakespeare works
- [ ] Implement attention mechanisms
- [ ] Add temperature sampling for diverse predictions
- [ ] Create beam search for better prediction accuracy
- [ ] Implement word embeddings (Word2Vec, GloVe)
- [ ] Add support for multi-word prediction
- [ ] Create REST API for production deployment

## 📜 License

This project is available as open source for educational purposes.

## 👤 Author

**Sukrit**
- GitHub: [@sukrit-89](https://github.com/sukrit-89)
- Project Link: [https://github.com/sukrit-89/WordPredictionLSTM](https://github.com/sukrit-89/WordPredictionLSTM)

## 🙏 Acknowledgments

- Shakespeare's "Hamlet" text from NLTK's Gutenberg corpus
- TensorFlow and Keras teams for the deep learning framework
- Streamlit for the easy-to-use web framework

---

**Note**: This is a pet project demonstrating the practical implementation of LSTM RNN for sequence prediction. The model's accuracy is limited by the small dataset size, but it successfully demonstrates the architecture and workflow of building a word prediction system.
