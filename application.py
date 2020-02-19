from flask import Flask
from flask import render_template, request
import pyjokes
import movie_recommender


app = Flask(__name__)


@app.route("/")
def index():

    return render_template(
        "main.html", title="Hello, World!", pyjoke=pyjokes.get_joke()
    )


@app.route("/recommendation", methods=["GET"])
def recommend():
    data = dict(request.args)
    new_user = [
        [data["movie1"], float(data["rating1"])],
        [data["movie2"], float(data["rating2"])],
        [data["movie3"], float(data["rating3"])],
    ]

    rec_NMF, rec_cosim = movie_recommender.recommend(new_user)

    return render_template(
        "recommendations.html",
        first=rec_cosim.iloc[0],
        second=rec_cosim.iloc[1],
        third=rec_cosim.iloc[2],
        last=rec_NMF.iloc[0],
    )


@app.route("/look_at_your_phone", methods=["GET"])
def look_at_my_phone():
    data = dict(request.args)
    movie_to_search = data["movie1"]
    movie_name, movie_rating, movie_tags = movie_recommender.show_movie_info(
        movie_to_search
    )
    return render_template(
        "look_at_your_phone.html",
        movie_name=movie_name,
        movie_rating=movie_rating,
        movie_tags=movie_tags,
    )
