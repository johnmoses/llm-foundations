# Large Language Model (LLM) for Education

Integrating LLMs with online learning systems. Core features include

1. LLM-Based Intent Detection

    - Utilizes a large language model (e.g., Hugging Face BLOOM) with few-shot prompting to classify user queries into predefined healthcare-related intents.
    - Returns intents in a structured format (JSON or parsed text) for reliable downstream handling.
    - Includes a fallback `"other"` intent for unrecognized queries

2. Structured Parameter Extraction

    - Uses the same LLM with targeted prompts to extract key parameters such as patient ID, name, age, and appointment date from natural language queries
    - Enables handlers to operate with precise, structured inputs rather than raw text

3. SQLite Local Database Integration

	- Stores and manages patient records, appointments, medications, and user feedback persistently
	- Supports CRUD operations on patient data and appointment scheduling
    - Provides medication reminders and tracks last visit timestamps

4. External API Integration

	- Connects to external healthcare APIs (e.g., FHIR servers) to fetch patient data when not available locally.
	- Enhances chatbot knowledge and real-world interoperability.

5. Robust Chatbot Response Generation

    - Maintains conversation history for context-aware responses.
    - Generates natural language replies using the LLM.
    - Supports fallback chat responses when intent is unclear.

6. Logging and Analytics

	- Logs all user queries, detected intents, and feedback to a file for monitoring and analysis.
	- Enables evaluation of intent detection accuracy and user satisfaction over time.

7. Command-Line Interface

	- Provides an interactive CLI for users to input natural language queries.
	- Routes queries through intent detection and parameter extraction to appropriate handlers.
	- Supports commands like adding patients, scheduling appointments, symptom checking, and feedback submission

8. Error Handling and User Guidance

	- Handles invalid inputs gracefully with prompts for clarification.
	- Provides fallback messages encouraging users to rephrase unclear queries.

## Requirements

```bash
pip install transformers torch
```

## How to Run

```bash
python app.py
```

## Sample queries

Course-related queries:
	•	“List all available courses.”
	•	“Show me the content of the Python Basics course.”
	•	“What courses are recommended for me based on my progress?”
	•	“How many courses have I completed so far?”
	•	Quiz-related queries:
	•	“Start the quiz for Data Science Intro.”
	•	“What is the next question in my current quiz?”
	•	“Give me feedback on my last quiz answer.”
	•	“Show me my quiz results for Python Basics.”
	•	Progress and user-related queries:
	•	“Update my progress to 75% for the Python Basics course.”
	•	“What is my current progress in Data Science Intro?”
	•	“Register me as a new user with username ‘alice’.”
	•	“How many users are enrolled in the Python Basics course?”
	•	General system queries:
	•	“Help me understand loops in Python.”
	•	“Explain the basics of data science.”
	•	“Show me a list of quizzes available for the Python Basics course.”