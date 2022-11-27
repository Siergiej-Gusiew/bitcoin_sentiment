from flask import Flask, request, render_template
from src.models.base_model import LogReg

# Flask instance
app = Flask(__name__)

# Model class instance
# model = LogReg()


@app.route("/", methods=["POST", "GET"])
def index():
    """Main form rendering"""
    if request.method == "POST":
        news_article = request.form["news"]
        label = LogReg().predict(news_article)
        return render_template("index.html", news=news_article, prediction=label)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    # for development set "debug=True" in app.run
    app.run(host="0.0.0.0", threaded=False, debug=True)
