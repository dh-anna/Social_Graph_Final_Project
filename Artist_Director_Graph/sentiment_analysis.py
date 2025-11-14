import statistics
import nltk


# Calculate a sentiment analysis for a given text and a dictionary of word sentiment scores statistics
def calculate_sentiment(text, word_scores):
    scores = []
    tokens = nltk.word_tokenize(text)
    total_tokens = len(tokens)
    scored_tokens = 0

    # Collect all scores
    for token in tokens:
        token_lower = token.lower()
        if token_lower in word_scores:
            scored_tokens += 1
            score = word_scores[token_lower]


            scores.append(score)

    used_tokens = len(scores)

    # Initialize result dictionary
    result = {
        'scores': scores
    }

    # Calculate statistics if we have scores
    if scores:
        result['mean'] = statistics.mean(scores)
        result['median'] = statistics.median(scores)
        result['variance'] = statistics.variance(scores) if len(scores) > 1 else 0.0

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