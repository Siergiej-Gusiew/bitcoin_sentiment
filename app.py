import os
import yaml
from flask import Flask, request, render_template
from src.models.base_model import LogReg


# Flask instance
app = Flask(__name__)


# Config LOAD:
directory_shift = ""

with open(directory_shift + "params.yaml") as conf_file:
    config = yaml.safe_load(conf_file)

base_model = config["model_output"]["base_model"]
vectorizer_dir = config["model_output"]["vectorizer"]


# Model class instance:
model = LogReg(config, base_model, vectorizer_dir)


@app.route("/", methods=["POST", "GET"])
def index():
    """Main form rendering"""
    if request.method == "POST":
        news_article = request.form["news"]
        category = model.predict(news_article)
        return render_template("index.html", news=news_article, prediction=category)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    # for development set "debug=True" in app.run
    # app.run(host="0.0.0.0", threaded=False, debug=True)

    # for run in Docker:
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
