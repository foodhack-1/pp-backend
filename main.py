import flask
from sklearn.externals import joblib
import numpy as np
import json
import jinja2
import sys
from settings import MODEL_PATH


MESSAGE_INVALID_FIELDS = jinja2.Template(
    '{{ \', \'.join(fields)}} {% if fields|length>1 %}are{% else %}is{% endif %} invalid'
)

app = flask.Flask(__name__)
clf = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return "Welcome", 200


@app.route("/predict", methods=["POST"])
def predict():
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
            response=json.dumps({'status': 'error', 'message': "You must provide at least user_vk or user_fb"}),
            status=400,
            mimetype='application/json')
    else:
        is_sporty = data.get('is_sporty', 0)
        is_single = data.get('is_single', 0)
        is_active = data.get('is_active', 0)
        is_employed = data.get('is_employed', 0)
        gender = data.get('gender', 0)
        is_healthy_eating = data.get('is_healthy_eating', 0)
        weight = data.get('weight', 0)



    try:
        result = clf.predict_proba(np.expand_dims(np.array([is_sporty, is_single, is_active, is_employed, gender,
                                                            is_healthy_eating, weight]), axis=1))
        print(f"Got verdict.")
        return app.response_class(
            response=json.dumps({"status": "ok", "predictions": result}),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        return app.response_class(
            response=json.dumps({"status": "error", "message": str(sys.exc_info()[1])}),
            status=200,
            mimetype='application/json'
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
