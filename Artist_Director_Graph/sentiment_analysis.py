import statistics
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from tqdm import tqdm
from transformers import pipeline


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


sentiment_pipeline = pipeline("sentiment-analysis")  # -1 for CPU, 0 for GPU


def calculate_sentiment(texts, batch_size=32):
    """
    Calculate sentiment analysis for a large array of texts using batching.

    Args:
        texts: List of text strings to analyze
        batch_size: Number of texts to process in each batch (default: 32)

    Returns:
        List of dictionaries containing sentiment results for each text.
        Each dict has 'label' (POSITIVE/NEGATIVE) and 'score' (confidence).
    """
    if not texts:
        return []

    # Filter out None or empty texts and keep track of original indices
    valid_texts = []
    valid_indices = []

    for i, text in enumerate(texts):
        if text and isinstance(text, str) and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)

    if not valid_texts:
        return [None] * len(texts)

    # Process texts in batches
    results = []
    for i in tqdm(range(0, len(valid_texts), batch_size)):
        batch = valid_texts[i:i + batch_size]
        try:
            # Pipeline handles batching internally
            batch_results = sentiment_pipeline(batch, truncation=True, max_length=512)
            results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add None for failed batch items
            results.extend([None] * len(batch))

    # Reconstruct full results list with None for invalid texts
    full_results = [None] * len(texts)
    for idx, result in zip(valid_indices, results):
        full_results[idx] = result

    return full_results


def calculate_sentiment_labmit(text, word_scores):
    lemmatizer = WordNetLemmatizer()
    labmit_scores = []
    lemmatized_docs = []
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    lemmatized_docs.append(' '.join(lemmatized))

    # Collect all scores
    for token in tokens:
        if token in word_scores:
            score = word_scores[token]
            labmit_scores.append(score)

    # Initialize result dictionary
    result = {
        'scores': labmit_scores
    }

    # Calculate statistics if we have scores
    if labmit_scores:
        result['labmit_mean'] = statistics.mean(labmit_scores)
        result['labmit_median'] = statistics.median(labmit_scores)

        sorted_scores = sorted(labmit_scores)

    else:
        # No scores available
        result['labmit_mean'] = None
        result['mean'] = None
        result['median'] = None
        result['variance'] = None

        result['percentile_25'] = None
        result['percentile_75'] = None

    return result

