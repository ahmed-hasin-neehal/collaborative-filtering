import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Ratings matrix (Items = rows, Users = columns)
data = {
    'User1': [5, 3, 4, np.nan, 2],
    'User2': [4, 5, np.nan, 3, 2],
    'User3': [4, np.nan, 5, 4, np.nan],
    'User4': [np.nan, 3, 4, 5, 3],
}
ratings_df = pd.DataFrame(data, index=['MovieA', 'MovieB', 'MovieC', 'MovieD', 'MovieE'])

# Step 2: Compute item-item similarity
# Fill missing values with 0 temporarily for similarity calculation
filled_items = ratings_df.fillna(0)
similarity_matrix = cosine_similarity(filled_items)
similarity_df = pd.DataFrame(similarity_matrix, index=ratings_df.index, columns=ratings_df.index)

# Step 3: Prediction logic — predict missing (item, user) values
def predict_rating_item_based(item, user):
    if not np.isnan(ratings_df.loc[item, user]):
        return ratings_df.loc[item, user]
    
    sim_scores = similarity_df.loc[item]
    weighted_sum = 0
    sim_total = 0
    
    for other_item in ratings_df.index:
        if other_item == item:
            continue
        rating = ratings_df.loc[other_item, user]
        if not np.isnan(rating):
            similarity = sim_scores[other_item]
            weighted_sum += similarity * rating
            sim_total += similarity
    
    return weighted_sum / sim_total if sim_total > 0 else np.nan

# Step 4: Fill in all missing values
predicted_df = ratings_df.copy()
for item in predicted_df.index:
    for user in predicted_df.columns:
        if pd.isna(predicted_df.loc[item, user]):
            predicted_df.loc[item, user] = predict_rating_item_based(item, user)

# Step 5: Show result
print("Final Predicted Ratings Matrix (Item-Item Collaborative Filtering):")
print(predicted_df.round(2))
