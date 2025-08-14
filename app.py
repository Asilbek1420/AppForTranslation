from flask import Flask, request, jsonify
import youtube_processor as yp

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_channel():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Please provide 'url' in JSON body"}), 400

    result = yp.process_channel(data["url"])
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YouTube Transcriber API is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
