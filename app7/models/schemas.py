from pydantic import BaseModel

class User(BaseModel):
    user_id: int
    username: str
    email: str | None = None

class Course(BaseModel):
    course_id: int
    title: str
    content: str

class Quiz(BaseModel):
    quiz_id: int
    course_id: int
    title: str

class Question(BaseModel):
    question_id: int
    quiz_id: int
    question_text: str
    choices: list[str]
    correct_answer: str
