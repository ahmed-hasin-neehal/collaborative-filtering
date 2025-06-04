import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'User1': [5, 3, 4, np.nan, 2],
    'User2': [4, 5, np.nan, 3, 2],
    'User3': [4, np.nan, 5, 4, np.nan],
    'User4': [np.nan, 3, 4, 5,3],
}
ratings_df = pd.DataFrame(data, index=['MovieA', 'MovieB', 'MovieC', 'MovieD','MovieE'])


filled = ratings_df.fillna(0)
similarity_matrix = cosine_similarity(filled.T)
similarity_df = pd.DataFrame(similarity_matrix, index=ratings_df.columns, columns=ratings_df.columns)

def predict_rating(item, user):
    if not np.isnan(ratings_df.loc[item, user]):
        return ratings_df.loc[item, user]

    sim_scores = similarity_df[user]
    weighted_sum = 0
    sim_total = 0
    
    for other_user in ratings_df.columns:
        if other_user == user:
            continue
        other_rating = ratings_df.loc[item, other_user]
        if not np.isnan(other_rating):
            sim = sim_scores[other_user]
            weighted_sum += sim * other_rating
            sim_total += sim
    
    return weighted_sum / sim_total if sim_total > 0 else np.nan

predicted_df = ratings_df.copy()
for item in predicted_df.index:
    for user in predicted_df.columns:
        if pd.isna(predicted_df.loc[item, user]):
            predicted_df.loc[item, user] = predict_rating(item, user)

print("Final Predicted Ratings Matrix:")
print(predicted_df.round(2))
