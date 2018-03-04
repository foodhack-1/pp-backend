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
    if not data:
        return app.response_class(
            response=json.dumps({'status': 'error', 'message': "You must provide at least one param"}),
            status=400,
            mimetype='application/json')
    else:
        breakfast_time = int(data.get("isBreakfast", 0))
        is_sporty = int(data.get('isSporty', 0))
        is_single = int(data.get('relation', 0))
        is_active = int(data.get('isActive', 0))
        is_employed = int(data.get('occupation', 0))
        gender = int(data.get('sex', 0))
        subscriptions = data.get('subscriptions', [])
        is_healthy_eating = int(bool(HEALTHY_PUBLICS.intersection(subscriptions)))
        weight = int(data.get('weight', 0))
        vector = [[is_sporty, is_single, is_active, is_employed, gender, is_healthy_eating, weight]]

    result = []
    try:
        pred = clf.predict_proba(vector).ravel()
        for i in range(len(pred)):
            if breakfast_time:
                population = inverted_dataset[i]
                result.extend(random.sample(population, int(pred[i] * len(population))))
            else:
                if i == 2 or i == 5:
                    population = inverted_dataset[i]
                else:
                    population = inverted_dataset[i + 6]
                result.extend(random.sample(population, int(pred[i] * len(population))))
    except Exception as e:
        print(str(sys.exc_info()[1]))
        return app.response_class(
            response=json.dumps({"status": "error", "result": result, "predictions": [0] * 7}),
            status=400,
            mimetype='application/json'
        )
    random.shuffle(result)
    return app.response_class(
        response=json.dumps({"status": "ok", "result": random.sample(result, 20), "predictions": list(pred)}),
        status=200,
        mimetype='application/json'
    )


@app.route("/page_<i>", methods=["GET", "POST"])
def request_for_page(i):
    data = flask.request.get_json()
    try:
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
        print("Page {} requested".format(i))
        return app.response_class(
            response=json.dumps({"status": "ok", "result": random.sample(inverted_dataset[index], 20)}),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"status": "error", "message": e.args}),
            status=200,
            mimetype='application/json'
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
