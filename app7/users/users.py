from db import get_connection


def add_user(username: str, email: str | None = None) -> int:
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email) VALUES (?, ?)", (username, email)
        )
        conn.commit()
        user_id = c.lastrowid
    except Exception:
        c.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_id = c.fetchone()[0]
    conn.close()
    return user_id
