# N-gram Language Model

## Overview
In Assigment 2, we implement unigram and bigram language model with add-k smoothing for training and evaluating text-based models. The model computes probabilities for sentences and evaluates perplexity on test data.

## Features
- **Train a language model** using unigrams and bigrams.
- **Predict log probabilities** for sentences using trained models.
- **Compute perplexity** to evaluate model performance.
- **Uses add-k smoothing** to handle unseen words.

## Installation & Run
1. Clone the repository:
   ```bash
   git clone https://github.com/smudigere/NgramLanguageModel
   cd NgramLanguageModel
   ```
2. Ensure you have Python 3 installed.
3. Run
    ```bash
   python3 language_modeling.py
   ```

## Usage
### Train the Model
```python
ngram_lm = NgramLanguageModel()
ngram_lm.train('samiam.train')
```

### Compute Perplexity
```python
print('Unigram Perplexity:', ngram_lm.test_perplexity('samiam.test', 'unigram'))
print('Bigram Perplexity:', ngram_lm.test_perplexity('samiam.test', 'bigram'))
```

## File Structure
- `language_modeling.py` - Main implementation of the language model.
- `samiam.train` - Training dataset.
- `samiam.test` - Test dataset.
- `readme.md` - Documentation.

## Bonus Solution
- Modified `NgramLanguageModel()` structure to include `self.total_unigrams = 0`.
- Instead of counting everytime from `self.unigram_counts`, initialized in the `train` function to be referenced later while calucalting perplexity.
- The above methodolgy improves test time.

## Sample Interaction
![Alt text](output.png?raw=true "Sample Interaction")

## Source(s)
- *Speech and Language Processing* – Daniel Jurafsky & James H. Martin
