"""
dispatcher.py

Handles JSON function call responses from the LLM by dispatching to backend handlers.
Supports functions like list_courses, view_course_content, update_progress, start_quiz,
quiz question handling, user registration, help, and general explanations.

Add or extend handlers as your application grows.
"""

import json
from courses.courses import get_courses, get_course_content
from quizzes.quizzes import get_quizzes_for_course, get_questions_for_quiz
from users.users import add_user
from db import get_connection

# --- Handler functions ---

def handle_list_courses(params, user_id=None):
    courses = get_courses()
    if not courses:
        return "No courses are currently available."
    return "\n".join(f"{cid}: {title}" for cid, title in courses)

def handle_view_course_content(params, user_id=None):
    course_id = params.get("course_id")
    if course_id is None:
        return "Missing parameter: course_id."
    content = get_course_content(course_id)
    return content

def handle_list_quizzes(params, user_id=None):
    course_id = params.get("course_id")
    if course_id is None:
        return "Missing parameter: course_id."
    quizzes = get_quizzes_for_course(course_id)
    if not quizzes:
        return "No quizzes available for this course."
    return "\n".join(f"{qid}: {title}" for qid, title in quizzes)

def handle_start_quiz(params, user_id=None):
    course_id = params.get("course_id")
    if course_id is None:
        return "Missing parameter: course_id."
    quizzes = get_quizzes_for_course(course_id)
    if not quizzes:
        return "No quizzes available for this course."
    quiz_id, quiz_title = quizzes[0]
    return f"Starting quiz '{quiz_title}'."

def handle_get_quiz_questions(params, user_id=None):
    quiz_id = params.get("quiz_id")
    if quiz_id is None:
        return "Missing parameter: quiz_id."
    questions = get_questions_for_quiz(quiz_id)
    if not questions:
        return "No questions found for this quiz."
    formatted = []
    for q in questions:
        choices = "\n".join(f"{i+1}. {c}" for i, c in enumerate(q["choices"]))
        formatted.append(f"Q: {q['question_text']}\n{choices}")
    return "\n\n".join(formatted)

def handle_submit_quiz_answer(params, user_id=None):
    # Placeholder: Implement quiz session and answer checking logic
    return "Quiz answer submission feature is under development."

def handle_get_quiz_results(params, user_id=None):
    # Placeholder: Implement fetching and formatting quiz results
    return "Quiz results feature is under development."

def handle_update_progress(params, user_id=None):
    if user_id is None:
        return "User ID is required to update progress."
    course_id = params.get("course_id")
    progress = params.get("progress")
    if course_id is None or progress is None:
        return "Missing parameters: course_id and/or progress."
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO progress (user_id, course_id, progress) VALUES (?, ?, ?)
        ON CONFLICT(user_id, course_id) DO UPDATE SET progress=excluded.progress
    ''', (user_id, course_id, progress))
    conn.commit()
    conn.close()
    return f"Progress updated to {progress}% for course ID {course_id}."

def handle_get_user_progress(params, user_id=None):
    if user_id is None:
        return "User ID is required to get progress."
    course_id = params.get("course_id")
    if course_id is None:
        return "Missing parameter: course_id."
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT progress FROM progress WHERE user_id = ? AND course_id = ?", (user_id, course_id))
    result = c.fetchone()
    conn.close()
    if result:
        return f"Your progress in course {course_id} is {result[0]}%."
    else:
        return "No progress found for this course."

def handle_register_user(params, user_id=None):
    username = params.get("username")
    if not username:
        return "Missing parameter: username."
    new_user_id = add_user(username)
    return f"User '{username}' registered with user ID {new_user_id}."

def handle_help(params, user_id=None):
    return ("You can ask me to list courses, start quizzes, check your progress, "
            "or explain course topics. How can I assist you today?")

def handle_general_explanation(params, user_id=None):
    topic = params.get("topic")
    if not topic:
        return "Please specify a topic you'd like explained."
    # Integrate your LLM explanation or knowledge base here
    return f"Here is a simple explanation of {topic}: [Explanation goes here]"

# --- Dispatch table ---

DISPATCH_TABLE = {
    "list_courses": handle_list_courses,
    "view_course_content": handle_view_course_content,
    "list_quizzes": handle_list_quizzes,
    "start_quiz": handle_start_quiz,
    "get_quiz_questions": handle_get_quiz_questions,
    "submit_quiz_answer": handle_submit_quiz_answer,
    "get_quiz_results": handle_get_quiz_results,
    "update_progress": handle_update_progress,
    "get_user_progress": handle_get_user_progress,
    "register_user": handle_register_user,
    "help": handle_help,
    "general_explanation": handle_general_explanation,
}

# --- Dispatcher function ---

def handle_function_call(response_json_str, user_id=None):
    """
    Dispatches the LLM function call JSON to the appropriate handler function.
    
    Parameters:
        response_json_str (str): JSON string from LLM specifying function and parameters.
        user_id (int, optional): Current user ID for context-dependent functions.
    
    Returns:
        str: User-friendly response string.
    """
    try:
        data = json.loads(response_json_str)
        func_name = data.get("function")
        params = data.get("parameters", {})

        handler = DISPATCH_TABLE.get(func_name)
        if handler:
            return handler(params, user_id=user_id)
        else:
            return f"Unknown function '{func_name}'. Please try rephrasing your request."

    except json.JSONDecodeError:
        return "Sorry, I couldn't understand the response."

    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"
