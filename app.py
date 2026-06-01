from flask import Flask, request, jsonify, render_template
from src.inference import predict

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_flower():
    data = request.get_json(force=True)
    features = data.get("features")
    if features is None:
        return jsonify({"error": "features field is required"}), 400

    try:
        prediction = int(predict(features))
        return jsonify({"prediction": prediction})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
