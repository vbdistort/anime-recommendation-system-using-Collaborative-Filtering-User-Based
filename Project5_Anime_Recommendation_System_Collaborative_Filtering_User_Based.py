# # Introduction

# In user-based collaborative filtering:
# - users are deemed similar if they like similar items
# - we first discover which users are similar
# - then recommend items that other similar users like

# Sunny likes paintings by Monet, Picasso and Dali.
# Ted likes paintings by Monet and Picasso.
# Sunny and Ted are similar because they like some of the same artists.
# Sunny likes Dali but Ted has never seen a Dali painting.
# So let's recommend Dali to Ted.

import pandas as pd
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import operator

animes = pd.read_csv(r'G:\Projects\dataset\anime-recommendations-database\anime.csv')
ratings = pd.read_csv(r'G:\Projects\dataset\anime-recommendations-database\rating.csv')

print("anime csv shape : ", animes.shape)
print(animes.head())

print("\n", "ratings csv shape : ", ratings.shape)
print(ratings.head())

ratings = ratings[ratings['rating'] != -1]

print("\n", "ratings csv shape after -1 ratings removed: ", ratings.shape)
print(ratings.head())

# Data exploration

print("\nno. of unique users : ", len(ratings['user_id'].unique()))
print("no. of unique animes : ", len(animes['anime_id'].unique()))

ratings_per_user = ratings.groupby('user_id')['rating'].count()
ratings_per_user = ratings_per_user.to_frame()
print(ratings_per_user.columns, "\n", ratings_per_user.head())
print("\nAvg. no. of ratings per user :",statistics.mean(ratings_per_user['rating']))

plt.hist(ratings_per_user['rating'], 20, (0,500))
plt.title("distribution of ratings per user")
plt.xlabel("no. of ratings ->")
plt.ylabel("no. of users ->")
plt.show(block=False)

ratings_per_anime = ratings.groupby('anime_id')['rating'].count()
ratings_per_anime = ratings_per_anime.to_frame()
print(ratings_per_anime.columns, "\n", ratings_per_anime.head())
print("\nAvg. no. of ratings per anime :",statistics.mean(ratings_per_anime['rating']))

plt.hist(ratings_per_anime['rating'], 20, (0,2500))
plt.title("distribution of ratings per anime")
plt.xlabel("no. of ratings ->")
plt.ylabel("no. of animes ->")
plt.show(block=False)

ratings_per_users = ratings_per_user[ratings_per_user['rating'] >= 500] # 1365 users left
ratings_per_anime = ratings_per_anime[ratings_per_anime['rating'] >= 1000] # 9591 animes left

print(ratings_per_user)
print(ratings_per_anime)

# build a list of user_ids to keep
prolific_users = ratings_per_users.index.tolist()
# build a list of anime_ids to keep
popular_anime = ratings_per_anime.index.tolist()

# print(prolific_users)
# print(popular_anime)
filtered_ratings = ratings[ratings.anime_id.isin(popular_anime)]
filtered_ratings = ratings[ratings.user_id.isin(prolific_users)]
print(filtered_ratings.shape)

matrix_userid_animeid_with_value_ratings = filtered_ratings.pivot_table(index='user_id', columns='anime_id', values='rating')
# replace NaN values with 0
matrix_userid_animeid_with_value_ratings = matrix_userid_animeid_with_value_ratings.fillna(0)

print(matrix_userid_animeid_with_value_ratings.head())

# Input from user
# current_user_id = input("enter current user id : ")
# current_user_id = int(current_user_id)

# num_of_animes_recommend = input("enter no. of animes to be recommended : ")
# num_of_animes_recommend = int(num_of_animes_recommend)

# Function to find the most similar users to the current_user using cosine similarity and find the 'num_of_similar_user' most similar users.

def get_similar_userids (user_id = 226, matrix = matrix_userid_animeid_with_value_ratings, num_of_similar_user = 3) :
    current_user_df = matrix[matrix.index == user_id] # shape = 1 x 9591
    other_users_df = matrix[matrix.index != user_id] # shape = 1364 x 9591
    print(other_users_df)

    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(current_user_df, other_users_df)[0].tolist() # cosine similarity = cos(theta) = A.B/|A||B|, that is, the dot product of 2 matrices
    # similarities = pd.DataFrame(similarities)

    indices = other_users_df.index.tolist()

    # create key/values pairs of user index and their similarity
    index_similarity_map = dict(zip(indices, similarities))
    # print(index_similarity_map)

    # sort by similarity score
    index_similarity_map = sorted(index_similarity_map.items(), key=lambda x: x[1], reverse=True)
    # print(index_similarity_map)

    # take top 'num_of_similar_users' from this map
    index_similarity_map = index_similarity_map[:num_of_similar_user]
    # top_similar_users = [u[0] for u in index_similarity_map]

    return index_similarity_map

# index_similarity_map = get_similar_userids(current_user_id, matrix_userid_animeid_with_value_ratings)
index_similarity_map = get_similar_userids()

def get_recommendations(user_id = 226, similarity_map = index_similarity_map, matrix = matrix_userid_animeid_with_value_ratings, num_of_recommendations = 5):
    # load vectors for similar users
    similar_user_indices = [u[0] for u in similarity_map]
    similar_users_vector = matrix[matrix.index.isin(similar_user_indices)]

    # calculate avg ratings for animes across the 3 similar users
    similar_users_vector = similar_users_vector.mean(axis=0)
    
    # convert to dataframe
    similar_users_df = pd.DataFrame(similar_users_vector, columns=['mean_rating'])

    current_user_df = matrix[matrix.index == user_id].transpose() #  transposed
    current_user_df.columns = ['rating']
    current_user_df = current_user_df[current_user_df['rating'] == 0] # only unseen animes required

    # generate a list of animes the user has not seen
    animes_unseen = current_user_df.index.tolist()
    # filter avg ratings of similar users for only anime the current user has not seen
    similar_users_df = similar_users_df[similar_users_df.index.isin(animes_unseen)]

    # order the dataframe
    similar_users_df = similar_users_df.sort_values(by=['mean_rating'], ascending=False)

    top_anime_indices = (similar_users_df.head(num_of_recommendations)).index.tolist()

    top_anime_list = animes[animes['anime_id'].isin(top_anime_indices)]
    print(top_anime_list)
    return top_anime_list

# get_recommendations(current_user_id, num_of_animes_recommend)
get_recommendations()