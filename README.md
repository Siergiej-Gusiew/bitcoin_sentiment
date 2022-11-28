# Bitcoin news sentiment

Flask-ML application to predict the sentiment of Bitcoin news

## Local deployment
------------
1. Unzip the repo locally and open a terminal in the repo's directory
2. (optionally) Inslall Docker if you don't have it: [Link](https://docs.docker.com/engine/install/)
3. Run the following in a terminal to build a Docker image:

        docker build -t bitcoin_sentiment .
4. Run the following in a terminal to run an application in a Docker container:

        docker run --rm -it -p 5000:5000 bitcoin_sentiment

## Project Organization
------------

    ├── README.md          <- The top-level README with the deployment instuctions
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── libs.txt           <- A list of installed libraries (deployment)
    │
    ├── libs-dev.txt       <- A list of installed libraries (dev)
    |
    ├── requirements.txt   <- The deployment requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── requirements-dev.txt   <- The requirements file for reproducing the project (dev)
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── Dockerfile         <- A script with instructions to build the project Docker image
    |
    ├── app.py             <- Flask application script
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
