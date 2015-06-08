# last updated 2015-06-04 Tong Shu Li
import pandas as pd

def confidence_skew_score(data):
    """
    Confidence skew score is defined as how the confidence of the
    crowd deviates from the "confident" choice.

    A score of 1.0 means every worker chose "very confident".
    A score of -1.0 means every worker chose "not very confident".

    The score is normalized by the number of workers. Basically,
    a positive score means people were overally very confident,
    with the magnitude of the score telling you what the distribution
    looks like.
    """
    mapping = {
        "very_confident": 1,
        "confident": 0,
        "not_very_confident": -1
    }

    return sum(map(lambda v: mapping[v], list(data["user_confidence"]))) / len(data)

def aggregate_votes(column_name, data_frame):
    """
    Given all of the human responses for one work unit,
    aggregates the results based on the column you give it.

    For each possible choice, it calculates:
        1. Total number of votes
        2. Confidence score

    Returns an unsorted data frame of the results.

    Confidence score is the sum of the trust score of
    individual workers.
    """
    temp = []

    for value, group in data_frame.groupby(column_name):
        conf_score = sum(group["_trust"])
        num_votes = len(group)

        conf_skew_score = confidence_skew_score(group)

        temp.append([value, conf_score, num_votes, conf_skew_score])

    return pd.DataFrame(temp, columns = [column_name, "conf_score", "num_votes", "conf_skew_score"])
