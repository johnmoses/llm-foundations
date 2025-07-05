from db import get_connection

def get_courses():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT course_id, title FROM courses")
    courses = c.fetchall()
    conn.close()
    return courses

def get_course_content(course_id: int) -> str:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT content FROM courses WHERE course_id = ?", (course_id,))
    content = c.fetchone()
    conn.close()
    return content[0] if content else "Course not found."
