import flask
from sklearn.externals import joblib
import numpy as np
import json
import jinja2
import sys
import random
import imblearn
from collections import defaultdict
from settings import MODEL_PATH, HEALTHY_PUBLICS


MESSAGE_INVALID_FIELDS = jinja2.Template(
    '{{ \', \'.join(fields)}} {% if fields|length>1 %}are{% else %}is{% endif %} invalid'
)

app = flask.Flask(__name__)
clf = joblib.load(MODEL_PATH)
dataset = json.load(open("dataset.json"))
inverted_dataset = defaultdict(list)
for d in dataset['items']:
    # title, time, photo, category, ingredients, instructions
    inverted_dataset[d['category']].append(d)


@app.route("/")
def index():
    return "Welcome", 200


def predict(vector):
    try:
        result = clf.predict_proba(vector)
        print("Got verdict.")
        return result

    except Exception as e:
        return app.response_class(
            response=json.dumps({"status": "error", "message": str(sys.exc_info()[1])}),
            status=200,
            mimetype='application/json'
        )


@app.route("/recommend", methods=["POST"])
def recommend():
    """
        POST http://0.0.0.0:9999/predict
        json={"is_sporty": int, "is_single": int, "is_active": int, is_employed: int, "gender": int,
        "is_healthy_eating":int, "weight": int}
        :return:
        [predictions list]: list
        """
    data = flask.request.get_json()
    vector = np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 0]), axis=1)
    if not data:
        return app.response_class(
            response=json.dumps({'status': 'error', 'message': "You must provide at least one param"}),
            status=400,
            mimetype='application/json')
    else:
        breakfast_time = data.get("isBreakfast", 0)
        is_sporty = data.get('isSporty', 0)
        is_single = data.get('relation', 0)
        is_active = data.get('isActive', 0)
        is_employed = data.get('occupation', 0)
        gender = data.get('sex', 0)
        subscriptions = data.get('subscriptions', [])
        is_healthy_eating = bool(HEALTHY_PUBLICS.intersection(subscriptions))
        weight = data.get('weight', 0)

        vector = np.expand_dims(np.array([is_sporty, is_single, is_active, is_employed, gender, is_healthy_eating,
                                          weight]), axis=1)
    pred = predict(vector)
    result = []
    for i in range(len(pred)):
        if breakfast_time:
            population = inverted_dataset[i + 6]
            result.extend(random.sample(population, pred[i] * len(population)))
        else:
            population = inverted_dataset[i + 6]
            result.extend(random.sample(population, pred[i] * len(population)))
    random.shuffle(result)
    return app.response_class(
        response=json.dumps({"status": "ok", "result": result, "predictions": list(pred)}),
        status=200,
        mimetype='application/json'
    )


@app.route("/page_<i>", methods=["GET", "POST"])
def request_for_page(i):
    data = flask.request.get_json()
    i = int(i)
    index = i
    if data:
        if not data.get("isBreakfast", 0):
            index = i
        else:
            if i == 2 or i == 5:
                index = i
            else:
                index = i + 6
    return app.response_class(
        response=json.dumps({"status": "ok", "result": inverted_dataset[index]}),
        status=200,
        mimetype='application/json'
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
