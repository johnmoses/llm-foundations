# Large Language Model (LLM) for Healthcare

Integrating LLMs with healthcare systems. Core features include

1. LLM-Based Intent Detection

    Utilizes a large language model (e.g., Hugging Face BLOOM) with few-shot prompting to classify user queries into predefined healthcare-related intents.
    Returns intents in a structured format (JSON or parsed text) for reliable downstream handling.
    Includes a fallback `"other"` intent for unrecognized queries.
2. Structured Parameter Extraction
    
    Uses the same LLM with targeted prompts to extract key parameters such as patient ID, name, age, and appointment date from natural language queries.
	Enables handlers to operate with precise, structured inputs rather than raw text.
3. SQLite Local Database Integration
	•	Stores and manages patient records, appointments, medications, and user feedback persistently.
	•	Supports CRUD operations on patient data and appointment scheduling.
	•	Provides medication reminders and tracks last visit timestamps.
4. External API Integration
	•	Connects to external healthcare APIs (e.g., FHIR servers) to fetch patient data when not available locally.
	•	Enhances chatbot knowledge and real-world interoperability.
    5. Robust Chatbot Response Generation
	•	Maintains conversation history for context-aware responses.
	•	Generates natural language replies using the LLM.
	•	Supports fallback chat responses when intent is unclear.
6. Logging and Analytics
	•	Logs all user queries, detected intents, and feedback to a file for monitoring and analysis.
	•	Enables evaluation of intent detection accuracy and user satisfaction over time.
7. Command-Line Interface
	•	Provides an interactive CLI for users to input natural language queries.
	•	Routes queries through intent detection and parameter extraction to appropriate handlers.
	•	Supports commands like adding patients, scheduling appointments, symptom checking, and feedback submission.
8. Error Handling and User Guidance
	•	Handles invalid inputs gracefully with prompts for clarification.
	•	Provides fallback messages encouraging users to rephrase unclear queries.

## Requirements

```bash
pip install transformers torch
```

## How to Run

```bash
python app.py
```

## Sample queries

Add Patient
“Add a new patient named John aged 45”
“Please register patient Alice who is 30 years old”

Get patient info:
“Show me the details of patient 3”
“Get patient information for ID 5”

Update conditions:
“Update conditions for patient 2 to diabetes and hypertension”
“Add asthma to patient 4’s conditions”

Update medications:
“Update medications for patient 1 to metformin and lisinopril”
“Add aspirin to patient 3’s medications”

Schedule appointment:
“Schedule an appointment for patient 2 on 2025-07-10 15:30”
“Book a doctor visit for patient 5 next Monday at 10 AM”

Medication reminders:
“Show medication reminders for patient 1”
“What meds should patient 4 take?”

Symptom check:
“I have a fever and cough”
“Feeling headache and nausea since yesterday”

Feedback:
“I want to give feedback for patient 3 rating 5”
“Rate the last response as 4 for patient 1”

Chat/general:
“Hello, how can you help me?”
“What is the best way to stay healthy?”

/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
