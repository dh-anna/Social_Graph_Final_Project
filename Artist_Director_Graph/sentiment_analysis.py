import statistics
from nltk import word_tokenize, pos_tag, WordNetLemmatizer


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

def duck():
    print('Duck')

# Calculate a sentiment analysis for a given text and a dictionary of word sentiment scores statistics
def calculate_sentiment(text, word_scores):
    scores = []
    lemmatizer = WordNetLemmatizer()

    lemmatized_docs = []
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    lemmatized_docs.append(' '.join(lemmatized))

    # Collect all scores
    for token in tokens:

        if token in word_scores:
            score = word_scores[token]
            scores.append(score)

    # Initialize result dictionary
    result = {
        'scores': scores
    }

    # Calculate statistics if we have scores
    if scores:
        result['mean'] = statistics.mean(scores)
        result['median'] = statistics.median(scores)
        result['variance'] = statistics.variance(scores) if len(scores) > 1 else 0.0
        result['min'] = min(scores)
        result['max'] = max(scores)

        sorted_scores = sorted(scores)
        result['percentile_25'] = statistics.quantiles(sorted_scores, n=4)[0] if len(scores) >= 2 else sorted_scores[0]
        result['percentile_75'] = statistics.quantiles(sorted_scores, n=4)[2] if len(scores) >= 2 else sorted_scores[0]
    else:
        # No scores available
        result['mean'] = None
        result['median'] = None
        result['variance'] = None

        result['percentile_25'] = None
        result['percentile_75'] = None

    return result

