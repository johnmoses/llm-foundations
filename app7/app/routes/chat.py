from flask import Blueprint, request, jsonify, render_template, session
from flask_login import login_required
from ..services.llama_service import generate_chat_response

chat_bp = Blueprint('chat', __name__, template_folder='../templates')

def get_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def save_chat_turn(user_msg, ai_msg):
    history = get_chat_history()
    history.append({'user': user_msg, 'ai': ai_msg})
    if len(history) > 10:
        history.pop(0)
    session['chat_history'] = history

@chat_bp.route('/', methods=['GET'])
@login_required
def chat_page():
    return render_template('chat.html')

@chat_bp.route('/message', methods=['POST'])
@login_required
def chat_message():
    user_message = request.json.get('message', '')
    history = get_chat_history()

    ai_response = generate_chat_response(history, user_message)

    save_chat_turn(user_message, ai_response)

    return jsonify({'response': ai_response})
