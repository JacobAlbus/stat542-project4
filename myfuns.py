import pandas as pd
import requests
import numpy as np

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

# data = pd.read_csv("data/Rmat.csv")

def grab_count(row):
        if row["movie_id"] in ratings_count:
            return ratings_count[row["movie_id"]]
        else:
            return 0
        
def grab_average_rating(row):
    reviews = ratings_data[ratings_data["movie_id"] == row["movie_id"]]
    if reviews.shape[0] == 0:
        return 0
    return reviews["Rating"].mean()

ratings_data = pd.read_csv("data/ratings.dat", sep=":", header=None)
ratings_data = ratings_data.drop([1, 3, 5], axis=1)
ratings_data = ratings_data.rename(columns={0 : "UserID", 2 : "movie_id", 4 : "Rating", 6 : "Timestamp"})
ratings_count = ratings_data["movie_id"].value_counts()

movies["Ratings_Count"] = movies.apply(grab_count, axis=1)
movies["Avg_Rating"] = movies.apply(grab_average_rating, axis=1)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies():
    return movies.head(100)

# def ibcf(newuser):
#     num_top_movies = 10
#     similarities = pd.read_csv("data/similarity_30.csv").fillna(0)
#     similarities = np.array(similarities.drop(["movie_ids"], axis=1))

#     newuser = np.array(newuser).flatten()
#     predictions = np.full(newuser.shape, np.nan) 

#     unrated_movies = np.where(np.isnan(newuser))[0]
#     for l in unrated_movies:

#         neighborhood = similarities[l,:]
        
#         top_similar_indices = ~np.isnan(neighborhood)
#         top_similarities = neighborhood[top_similar_indices]

#         rated_by_user = ~np.isnan(newuser[top_similar_indices])
        
#         if rated_by_user.any():
#             ratings = newuser[top_similar_indices][rated_by_user]

#             weighted_ratings_sum = np.dot(top_similarities[rated_by_user], ratings)

#             sum_of_similarities = np.sum(np.abs(top_similarities[rated_by_user]))

#             prediction = weighted_ratings_sum / sum_of_similarities if sum_of_similarities != 0 else np.nan
#             predictions[l] = prediction
    
#     sorted_indices = np.argsort(-predictions, kind = 'stable')
#     valid_predictions_count = np.count_nonzero(np.isnan(predictions))

    
#     top_rated_indices = sorted_indices[:10]

#     if valid_predictions_count >= num_top_movies:
#         top_rated_indices = sorted_indices[:10]
#         top_movies = data.columns[top_rated_indices].tolist()
#         top_predictions = predictions[top_rated_indices].tolist()
#         return top_movies, top_predictions

#     # If we don't have enough ratings, then fill in rest of top 10 movies
#     # with top movies from randomly selected genre
#     else:
#         top_rated_indices = sorted_indices[:valid_predictions_count]

#         top_movies = data.columns[top_rated_indices].tolist()
#         top_predictions = predictions[top_rated_indices].tolist()
        
#         genre = np.random.choice(genres)
#         top_genre_movies = list(get_popular_movies(genre)["movieID"])
#         top_genre_predictions = [0 for _ in range(num_top_movies)]
        
#         top_movies.extend(top_genre_movies[valid_predictions_count:])
#         top_predictions.extend(top_genre_predictions[valid_predictions_count:])

#         return top_movies, top_predictions


def get_recommended_movies(new_user_ratings):
    return movies.head(10)

# def get_recommended_movies(new_user_ratings):
#     user = data.loc["u1181"].copy()
#     for col in data.columns:
#         user[col] = np.NaN

#     for movie_id in new_user_ratings:
#         full_id = f"m{movie_id}"
#         user[full_id] = new_user_ratings[movie_id]
        
#     top_movies, predictions = ibcf(user)
    
#     ids = []
#     for movie_id in top_movies:
#         ids.append(int(movie_id[1:]))
#     return movies.iloc[ids]

def get_popular_movies(genre: str):
    movies_in_genre = movies[movies["genres"].str.contains(genre)].copy()

    movies_in_genre['Ratings_Count'] = (movies_in_genre['Ratings_Count'] - movies_in_genre['Ratings_Count'].mean()) / movies_in_genre['Ratings_Count'].std()
    movies_in_genre['Avg_Rating'] = (movies_in_genre['Avg_Rating'] - movies_in_genre['Avg_Rating'].mean()) / movies_in_genre['Avg_Rating'].std()
    movies_in_genre["Recommendation_Rating"] = movies_in_genre['Ratings_Count'] + movies_in_genre['Avg_Rating']
    
    movies_in_genre = movies_in_genre.sort_values(by=["Recommendation_Rating"], ascending=False)
    
    num_picks = 10
    return movies_in_genre.iloc[:num_picks]
