# app/app.py

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from services.chat_service import ChatService

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "FRONTEND" / "dist"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST),
    static_url_path="/",
)
CORS(app)

chatbot = ChatService()

@app.route("/")
def index():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return send_from_directory(app.static_folder, "index.html")
    return (
        "React build not found. Run `npm install` and `npm run build` in "
        "EmpowerHer_Chatbot/FRONTEND, then start the backend.",
        404,
    )


@app.route("/<path:path>")
def static_proxy(path: str):
    file_path = FRONTEND_DIST / path
    if file_path.exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")

    result = chatbot.generate_reply(message)

    return jsonify({
        "reply": result.reply,
        "emotions": result.emotions,
        "raw_emotions": result.raw_emotions,
        "topic": result.topic,
        "intent": result.intent,
        "kb_sources": result.kb_sources,
    })



if __name__ == "__main__":
    app.run(debug=True)
