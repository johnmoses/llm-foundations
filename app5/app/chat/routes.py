from flask import Blueprint, request, jsonify, render_template, session
from ..service import generate_response

chat_bp = Blueprint('chat_bp', __name__, template_folder='../templates/chat')

@chat_bp.route('/chat', methods=['GET'])
def chat_page():
    session['chat_history'] = []  # reset chat session on page load
    return render_template('chat/chatbot.html')

@chat_bp.route('/chat', methods=['POST'])
def chat_api():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({"response": "Please type a message."})

    history = session.get('chat_history', [])
    history.append({"role": "user", "content": user_message})

    ai_reply = generate_response(history)
    history.append({"role": "assistant", "content": ai_reply})

    session['chat_history'] = history
    return jsonify({"response": ai_reply})

