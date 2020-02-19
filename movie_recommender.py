# %%
from sqlalchemy import create_engine
from pprint import pprint
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

PLOT = False
# %% instantiate postgres
pguser = "postgres"
pgpw = "postgres"
pghost = "moviedb.cqtoauy8eqbm.us-east-2.rds.amazonaws.com"
pgport = 5432
pgdb = "moviedb"
conns = f"postgres://{pguser}:{pgpw}@{pghost}:{pgport}/{pgdb}"
db_pg = create_engine(conns, encoding="latin1", echo=False)

# %% Load dataframes from SQL
df_movies = pd.read_sql("SELECT * FROM movies", db_pg)
df_ratings = pd.read_sql("SELECT * FROM ratings", db_pg)
df_tags = pd.read_sql("SELECT * FROM tags", db_pg)

# %% Get the rating of each movie for each user, fill in 0 for empty entries

df_rating_user = df_ratings.groupby(["useId", "movieId"])["rating"].mean().unstack()
df_rating_user.fillna(df_rating_user.mean(), inplace=True)
# Non-Negative Matrix Factorization
model = NMF(n_components=30, init="random", random_state=10)
model.fit(df_rating_user)

Q = model.components_
P = model.transform(df_rating_user)

nR = np.dot(P, Q)
# Calculate the recommended movie for each user
df_predictions = pd.DataFrame(
    data=nR,
    index=df_rating_user.index,
    columns=df_movies[df_movies["movieid"].isin(df_rating_user.columns)]["title"],
)
recommended_movies = df_predictions.idxmax(axis=1)

# %%
def save_obj(obj, name):
    """saves object into pickle file """
    with open("out/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """load object from pickle file """
    with open("out/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


def to_vec(new_user):
    """takes the user's movie names and ratings and
    turn it into a vector"""
    movie_index = []
    # use fuzzywuzzy to extract the index of the closest match
    # to the user's movies
    for movie in new_user:
        movie_index.append(process.extractOne(movie[0], df_movies["title"])[2])
    rating_vec = np.zeros(df_rating_user.columns.shape)
    np.put(rating_vec, movie_index, new_user[:, 1])
    return rating_vec


def get_recommendation(new_user, Q):
    """ takes the user's movie names and ratings and 
    returns the recommended movie"""
    rating_vec = to_vec(new_user)
    user_P = model.transform(rating_vec.reshape(1, -1))
    user_nR = np.dot(user_P, Q)
    df_predictions = pd.DataFrame(
        data=user_nR,
        index=[0],
        columns=df_movies[df_movies["movieid"].isin(df_rating_user.columns)]["title"],
    )
    recommended_movies_user = df_predictions.idxmax(axis=1)
    return recommended_movies_user


def collaborative_filter(new_user, nR, recommended_movies, no_of_recommendations):
    """takes the user's movie names and ratings and
    returns no. of similar movies"""
    rating_vec = to_vec(new_user)
    user_P = model.transform(rating_vec.reshape(1, -1))
    user_nR = np.dot(user_P, Q)
    cos_sim = cosine_similarity(user_nR, nR)
    ranking = np.argsort(cos_sim)
    ranking = ranking.reshape(-1,)
    ranking = ranking[::-1]

    recommended_movies_user = recommended_movies.loc[ranking[:20]]
    recommended_movies_user.drop_duplicates(keep="first", inplace=True)
    return recommended_movies_user


def show_movie_info(movie_name):
    movie_name = process.extractOne(movie_name, df_movies["title"])[0]
    movie_id = df_movies.loc[df_movies["title"] == movie_name]["movieid"]
    movie_rating = np.round(
        np.mean(df_ratings[df_ratings["movieId"] == movie_id.iloc[0]]["rating"]), 2
    )
    movie_tags = list(set(df_tags[df_tags["movieId"] == movie_id.iloc[0]]["tag"]))
    return movie_name, movie_rating, movie_tags


# %% test for recommending 1 movie
def recommend(new_user):
    new_user = np.array(new_user)
    recommended_movies_user = get_recommendation(new_user, Q)
    recommended_movies_collab_filter = collaborative_filter(
        new_user, nR, recommended_movies, 10
    )
    return recommended_movies_user, recommended_movies_collab_filter


# %% plot heatmap of cosine similarity
if PLOT:
    result = cosine_similarity(nR, nR)
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(ax=ax, data=result)
    plt.ylim(0, nR.shape[0])
    plt.xlim(0, nR.shape[0])

