import sqlite3
import json
import time

DB_PATH = "education_system.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def clear_tables():
    conn = get_connection()
    c = conn.cursor()
    # Delete all existing data to avoid duplicates on re-run
    c.execute("DELETE FROM quiz_results")
    c.execute("DELETE FROM progress")
    c.execute("DELETE FROM chat_history")
    c.execute("DELETE FROM questions")
    c.execute("DELETE FROM quizzes")
    c.execute("DELETE FROM courses")
    c.execute("DELETE FROM users")
    conn.commit()
    conn.close()


def seed_users():
    users = [
        ("alice", "alice@example.com"),
        ("bob", "bob@example.com"),
        ("charlie", "charlie@example.com"),
    ]
    conn = get_connection()
    c = conn.cursor()
    c.executemany("INSERT INTO users (username, email) VALUES (?, ?)", users)
    conn.commit()
    conn.close()


def seed_courses():
    courses = [
        (
            "Python Basics",
            "Learn Python fundamentals including variables, loops, and functions.",
        ),
        (
            "Data Science Intro",
            "Introduction to data analysis, visualization, and machine learning.",
        ),
    ]
    conn = get_connection()
    c = conn.cursor()
    c.executemany("INSERT INTO courses (title, content) VALUES (?, ?)", courses)
    conn.commit()
    conn.close()


def seed_quizzes_and_questions():
    conn = get_connection()
    c = conn.cursor()

    # Get course IDs to link quizzes
    c.execute("SELECT course_id FROM courses WHERE title = ?", ("Python Basics",))
    python_course_id = c.fetchone()[0]
    c.execute("SELECT course_id FROM courses WHERE title = ?", ("Data Science Intro",))
    ds_course_id = c.fetchone()[0]

    # Insert quizzes
    quizzes = [
        (python_course_id, "Python Basics Quiz"),
        (ds_course_id, "Data Science Intro Quiz"),
    ]
    c.executemany("INSERT INTO quizzes (course_id, title) VALUES (?, ?)", quizzes)
    conn.commit()

    # Get inserted quiz IDs
    c.execute("SELECT quiz_id FROM quizzes WHERE title = ?", ("Python Basics Quiz",))
    python_quiz_id = c.fetchone()[0]
    c.execute(
        "SELECT quiz_id FROM quizzes WHERE title = ?", ("Data Science Intro Quiz",)
    )
    ds_quiz_id = c.fetchone()[0]

    # Insert questions for Python Basics Quiz
    python_questions = [
        {
            "quiz_id": python_quiz_id,
            "question_text": "What keyword is used to define a function in Python?",
            "choices": ["def", "func", "function", "define"],
            "correct_answer": "def",
        },
        {
            "quiz_id": python_quiz_id,
            "question_text": "Which data type is immutable?",
            "choices": ["list", "dict", "tuple", "set"],
            "correct_answer": "tuple",
        },
        {
            "quiz_id": python_quiz_id,
            "question_text": "What symbol is used for comments in Python?",
            "choices": ["//", "#", "/*", "--"],
            "correct_answer": "#",
        },
    ]

    # Insert questions for Data Science Intro Quiz
    ds_questions = [
        {
            "quiz_id": ds_quiz_id,
            "question_text": "Which library is commonly used for data manipulation in Python?",
            "choices": ["NumPy", "Pandas", "Matplotlib", "Seaborn"],
            "correct_answer": "Pandas",
        },
        {
            "quiz_id": ds_quiz_id,
            "question_text": "What does 'ML' stand for?",
            "choices": [
                "Machine Learning",
                "Model Logic",
                "Maximum Likelihood",
                "Meta Learning",
            ],
            "correct_answer": "Machine Learning",
        },
    ]

    # Helper function to insert questions
    def insert_questions(questions):
        for q in questions:
            c.execute(
                "INSERT INTO questions (quiz_id, question_text, choices, correct_answer) VALUES (?, ?, ?, ?)",
                (
                    q["quiz_id"],
                    q["question_text"],
                    json.dumps(q["choices"]),
                    q["correct_answer"],
                ),
            )
        conn.commit()

    insert_questions(python_questions)
    insert_questions(ds_questions)

    conn.close()


def seed_progress_and_results():
    # Example progress and quiz results for user Alice (assuming user_id=1)
    conn = get_connection()
    c = conn.cursor()

    # Get user_id for Alice
    c.execute("SELECT user_id FROM users WHERE username = ?", ("alice",))
    alice_id = c.fetchone()[0]

    # Get course and quiz IDs
    c.execute("SELECT course_id FROM courses WHERE title = ?", ("Python Basics",))
    python_course_id = c.fetchone()[0]
    c.execute("SELECT quiz_id FROM quizzes WHERE title = ?", ("Python Basics Quiz",))
    python_quiz_id = c.fetchone()[0]

    # Insert progress
    c.execute(
        """
        INSERT INTO progress (user_id, course_id, progress) VALUES (?, ?, ?)
        ON CONFLICT(user_id, course_id) DO UPDATE SET progress=excluded.progress
    """,
        (alice_id, python_course_id, 75),
    )

    # Insert quiz result
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        """
        INSERT INTO quiz_results (user_id, quiz_id, score, total_questions, passed, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (alice_id, python_quiz_id, 3, 3, True, timestamp),
    )

    conn.commit()
    conn.close()


def seed_all():
    print("Clearing existing data...")
    clear_tables()
    print("Seeding users...")
    seed_users()
    print("Seeding courses...")
    seed_courses()
    print("Seeding quizzes and questions...")
    seed_quizzes_and_questions()
    print("Seeding progress and quiz results...")
    seed_progress_and_results()
    print("Database seeding complete.")


if __name__ == "__main__":
    seed_all()
