# Transformer for English-to-French Translation

This project implements a simple Transformer model using PyTorch to perform English-to-French translation. The model is trained on a small dataset of sentence pairs and can translate user-input sentences after training.

## Features
- Implements a Transformer-based sequence-to-sequence model.
- Utilizes positional encoding for better sequence learning.
- Trains using a small dataset of English-to-French sentence pairs.
- Allows user input for real-time translation after training.
- Runs on GPU if available for faster computations.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/transformer-translation.git
   cd transformer-translation
   ```
2. Install dependencies:
   ```bash
   pip install torch torchtext
   ```
3. Ensure that you have a GPU available (optional but recommended).

## Usage
### Training the Model
Run the script to train the Transformer model:
```bash
python transformer.py
```
The model will be trained for a few epochs and saved to `transformer_model.pth`.

### Translating a Sentence
After training, you can enter an English sentence to get its French translation:
```bash
python transformer.py
```
Example:
```
Enter a sentence to translate: where are you from
Translation: d'où viens-tu
```

## Model Details
- **Encoder-Decoder Architecture**: Uses `nn.TransformerEncoder` and `nn.TransformerDecoder`.
- **Embedding Layer**: Converts input words into dense vector representations.
- **Positional Encoding**: Helps the model understand word order.
- **Cross-Entropy Loss**: Used for training the model.
- **Adam Optimizer**: Used for gradient updates.

## Future Improvements
- Increase dataset size for better translation accuracy.
- Implement beam search decoding for better output sequences.
- Add support for more language pairs.

## License
This project is open-source. The license details can be specified here.

