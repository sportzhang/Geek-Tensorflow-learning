import json
import requests

json_data = requests.get("https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json").json()


with open(r"H:\kevin\learning_projects\my-tf2-learning-note\chapter_7\imdb_word_index.json", encoding="utf-8", mode="w") as f:
    json.dump(json_data, f)
