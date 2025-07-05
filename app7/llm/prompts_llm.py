from llama_cpp import Llama

NL_TO_SQL_PROMPT = """
You are an assistant that translates natural language questions about an educational platform into valid SQLite SQL queries.

Only output the SQL query without any explanation.

Example:
Question: How many courses has user 'alice' completed?
SQL: SELECT COUNT(*) FROM progress p JOIN users u ON p.user_id = u.user_id WHERE u.username = 'alice' AND p.progress = 100;

Question: {user_question}
SQL:
"""

RECOMMENDATION_PROMPT = """
You are an educational advisor. Given the following user progress summary, recommend the next best course or topic for the user to study next.

User progress summary:
{progress_summary}

Available courses:
{course_list}

Recommendation:
"""

QUIZ_QUESTION_PROMPT = """
You are a quiz tutor. Present the following question with multiple choice answers clearly numbered.

Question:
{question_text}

Choices:
{choices}

Please answer by typing the number of your choice.
"""

QUIZ_FEEDBACK_PROMPT = """
User answered: {user_answer}

Correct answer: {correct_answer}

If the user's answer matches the correct answer, respond with "Correct!". Otherwise, respond with "Incorrect. The correct answer is {correct_answer}."
"""

COURSE_EXPLANATION_PROMPT = """
You are a helpful tutor. Explain the following course content in simple, clear language suitable for beginners:

{course_content}

Explanation:
"""

FUNCTION_CALLING_PROMPT = """
You are an educational assistant. When the user asks a question or makes a request, respond ONLY with a JSON object specifying the function to call and its parameters.

Available functions:

- list_courses(): Lists all courses.
- view_course_content(course_id): Shows content of a course.
- update_progress(course_id, progress): Updates user's progress.
- start_quiz(course_id): Starts a quiz for a course.

Example response:

{{
  "function": "view_course_content",
  "parameters": {{"course_id": 2}}
}}

User input: {user_input}

JSON response:
"""

def render_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)

class LLMWrapper:
    def __init__(self, model_path: str, n_threads: int = 8, n_gpu_layers: int = 0, n_ctx: int = 2048):
        self.llm = Llama(
            model_path=model_path,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7, top_k: int = 40, stop=None) -> str:
        stop = stop or ["</s>"]
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
            echo=False
        )
        return response['choices'][0]['text'].strip()

    def nl_to_sql(self, user_question: str) -> str:
        prompt = render_prompt(NL_TO_SQL_PROMPT, user_question=user_question)
        return self.generate(prompt)

    def recommend_courses(self, progress_summary: str, course_list: str) -> str:
        prompt = render_prompt(RECOMMENDATION_PROMPT, progress_summary=progress_summary, course_list=course_list)
        return self.generate(prompt)

    def quiz_question(self, question_text: str, choices: list) -> str:
        choices_text = "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(choices))
        prompt = render_prompt(QUIZ_QUESTION_PROMPT, question_text=question_text, choices=choices_text)
        return self.generate(prompt)

    def quiz_feedback(self, user_answer: str, correct_answer: str) -> str:
        prompt = render_prompt(QUIZ_FEEDBACK_PROMPT, user_answer=user_answer, correct_answer=correct_answer)
        return self.generate(prompt)

    def explain_course_content(self, course_content: str) -> str:
        prompt = render_prompt(COURSE_EXPLANATION_PROMPT, course_content=course_content)
        return self.generate(prompt)

    def function_calling(self, user_input: str) -> str:
        prompt = render_prompt(FUNCTION_CALLING_PROMPT, user_input=user_input)
        return self.generate(prompt)
