import json
from db import get_connection

def get_quizzes_for_course(course_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT quiz_id, title FROM quizzes WHERE course_id = ?", (course_id,))
    quizzes = c.fetchall()
    conn.close()
    return quizzes

def get_questions_for_quiz(quiz_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT question_id, question_text, choices, correct_answer FROM questions WHERE quiz_id = ?", (quiz_id,))
    questions = c.fetchall()
    conn.close()
    return [
        {
            "question_id": q[0],
            "question_text": q[1],
            "choices": json.loads(q[2]),
            "correct_answer": q[3]
        }
        for q in questions
    ]
