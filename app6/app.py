import json
import logging
import sqlite3
from datetime import datetime
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------- Configuration -----------

FHIR_API_BASE = "https://your-fhir-server.com"  # Replace with your FHIR server URL
FHIR_API_TOKEN = "your_api_token"  # Replace with your token or leave empty if none

MODEL_NAME = "bigscience/bloom-560m"  # Hugging Face model for LLM tasks

# ----------- Setup Logging -----------

logging.basicConfig(
    filename="chatbot_intent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_intent(user_query, detected_intent, actual_intent=None):
    logging.info(
        f"User Query: {user_query} | Detected Intent: {detected_intent} | Actual Intent: {actual_intent}"
    )


# ----------- Initialize SQLite DB -----------


def init_db(db_path="healthcare.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            conditions TEXT DEFAULT '',
            medications TEXT DEFAULT '',
            last_visit TEXT DEFAULT ''
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            datetime TEXT,
            confirmed INTEGER DEFAULT 0
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            query TEXT,
            response TEXT,
            rating INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    return conn


# ----------- Database Utility Functions -----------


def add_patient(conn, name, age):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO patients (name, age) VALUES (?, ?)", (name, age))
    conn.commit()
    return cursor.lastrowid


def get_patient(conn, patient_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, age, conditions, medications, last_visit FROM patients WHERE id = ?",
        (patient_id,),
    )
    return cursor.fetchone()


def update_patient_conditions(conn, patient_id, conditions):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE patients SET conditions = ?, last_visit = ? WHERE id = ?",
        (conditions, datetime.now().isoformat(), patient_id),
    )
    conn.commit()


def update_patient_medications(conn, patient_id, medications):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE patients SET medications = ?, last_visit = ? WHERE id = ?",
        (medications, datetime.now().isoformat(), patient_id),
    )
    conn.commit()


def schedule_appointment(conn, patient_id, dt_str):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO appointments (patient_id, datetime, confirmed) VALUES (?, ?, 1)",
        (patient_id, dt_str),
    )
    conn.commit()
    return cursor.lastrowid


def get_medication_reminders(conn, patient_id):
    patient = get_patient(conn, patient_id)
    if not patient or not patient[4]:
        return "No medications recorded."
    meds = patient[4].split(";")
    reminders = [
        f"Reminder: Take {med.strip()} as prescribed." for med in meds if med.strip()
    ]
    return "\n".join(reminders) if reminders else "No medications recorded."


def add_feedback(conn, patient_id, query, response, rating):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (patient_id, query, response, rating) VALUES (?, ?, ?, ?)",
        (patient_id, query, response, rating),
    )
    conn.commit()


# ----------- External API Integration -----------


def fetch_patient_from_fhir(patient_id):
    url = f"{FHIR_API_BASE}/Patient/{patient_id}"
    headers = {"Authorization": f"Bearer {FHIR_API_TOKEN}"} if FHIR_API_TOKEN else {}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": f"FHIR API request failed: {e}"}


# ----------- Symptom Checker -----------


def simple_symptom_checker(symptoms):
    symptoms = symptoms.lower()
    if "fever" in symptoms and "cough" in symptoms:
        return "You might have a common cold or flu. Please rest and stay hydrated."
    elif "headache" in symptoms and "nausea" in symptoms:
        return "These symptoms could indicate migraine or other conditions. Consider consulting a doctor."
    elif "chest pain" in symptoms:
        return "Chest pain can be serious. Please seek immediate medical attention."
    else:
        return "Symptoms unclear. Please provide more details or consult a healthcare professional."


# ----------- Load LLM Model -----------

print("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded.")


def llm_generate(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,       # optional: for more varied output
        temperature=0.7       # optional: controls randomness
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()


# ----------- Intent Detection -----------

INTENTS = [
    "add_patient",
    "get_patient",
    "update_conditions",
    "update_medications",
    "schedule_appointment",
    "medication_reminders",
    "symptom_check",
    "feedback",
    "chat",
    "other"
]

FEW_SHOT_INTENT_PROMPT = """
You are an intent classification assistant for a healthcare chatbot. Given a user query, classify it into one of these intents: {intents}.

Examples:
User query: "Add a new patient named John aged 45"
Intent: add_patient

User query: "Show me the details of patient 3"
Intent: get_patient

User query: "Schedule an appointment for patient 2 on 2025-07-10 15:30"
Intent: schedule_appointment

User query: "I have a fever and cough"
Intent: symptom_check

User query: "{query}"
Intent:
"""

def detect_intent_llm(user_query):
    prompt = FEW_SHOT_INTENT_PROMPT.format(intents=", ".join(INTENTS), query=user_query)
    response_text = llm_generate(prompt)

    # Robust parsing: extract first line with intent keyword
    lines = response_text.strip().splitlines()
    for line in lines:
        line_lower = line.lower()
        for intent in INTENTS:
            if intent in line_lower:
                return intent
    # Fallback to "other" if no intent found
    return "other"

# ----------- Parameter Extraction -----------

PARAM_EXTRACTION_PROMPT = """
Extract the following parameters from the user query if present: patient_id, name, age, appointment_date.

Return a JSON object with these keys. If a parameter is missing, set it to null.

User query: "{query}"

Respond only with a JSON object.
"""


def extract_parameters_llm(user_query):
    prompt = PARAM_EXTRACTION_PROMPT.format(query=user_query)
    response_text = llm_generate(prompt)
    try:
        params = json.loads(response_text)
    except json.JSONDecodeError:
        params = {}
    return params


# ----------- Chatbot Response -----------

conversation_history = []


def generate_llm_response(user_input):
    global conversation_history
    conversation_history.append(f"User: {user_input}")
    context = "\n".join(conversation_history[-20:])
    inputs = tokenizer(
        context, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    outputs = model.generate(
        **inputs, max_length=150, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    conversation_history.append(f"Bot: {response}")
    return response


# ----------- Intent Handlers -----------


def handle_add_patient(query, conn):
    params = extract_parameters_llm(query)
    name = params.get("name") or input("Enter patient name: ")
    age = params.get("age")
    if not age:
        try:
            age = int(input("Enter patient age: "))
        except ValueError:
            return "Invalid age input."
    pid = add_patient(conn, name, int(age))
    return f"Patient '{name}' added with ID {pid}."


def handle_get_patient(query, conn):
    params = extract_parameters_llm(query)
    pid = params.get("patient_id")
    if not pid:
        try:
            pid = int(input("Enter patient ID: "))
        except ValueError:
            return "Invalid patient ID."
    else:
        pid = int(pid)
    patient = get_patient(conn, pid)
    if patient:
        return f"ID: {patient[0]}, Name: {patient[1]}, Age: {patient[2]}, Conditions: {patient[3]}, Medications: {patient[4]}, Last Visit: {patient[5]}"
    fhir_data = fetch_patient_from_fhir(pid)
    if "error" in fhir_data:
        return fhir_data["error"]
    name = fhir_data.get("name", [{}])[0].get("text", "Unknown")
    birthdate = fhir_data.get("birthDate", "Unknown")
    return f"FHIR API: Name: {name}, Birthdate: {birthdate}"


def handle_update_conditions(query, conn):
    params = extract_parameters_llm(query)
    pid = params.get("patient_id")
    if not pid:
        try:
            pid = int(input("Enter patient ID: "))
        except ValueError:
            return "Invalid patient ID."
    else:
        pid = int(pid)
    conditions = input("Enter conditions (semicolon separated): ")
    update_patient_conditions(conn, pid, conditions)
    return "Patient conditions updated."


def handle_update_medications(query, conn):
    params = extract_parameters_llm(query)
    pid = params.get("patient_id")
    if not pid:
        try:
            pid = int(input("Enter patient ID: "))
        except ValueError:
            return "Invalid patient ID."
    else:
        pid = int(pid)
    medications = input("Enter medications (semicolon separated): ")
    update_patient_medications(conn, pid, medications)
    return "Patient medications updated."


def handle_schedule_appointment(query, conn):
    params = extract_parameters_llm(query)
    pid = params.get("patient_id")
    dt_str = params.get("appointment_date")
    if not pid:
        try:
            pid = int(input("Enter patient ID: "))
        except ValueError:
            return "Invalid patient ID."
    else:
        pid = int(pid)
    if not dt_str:
        dt_str = input("Enter appointment datetime (YYYY-MM-DD HH:MM): ")
    appt_id = schedule_appointment(conn, pid, dt_str)
    return f"Appointment scheduled with ID {appt_id}."


def handle_medication_reminders(query, conn):
    params = extract_parameters_llm(query)
    pid = params.get("patient_id")
    if not pid:
        try:
            pid = int(input("Enter patient ID: "))
        except ValueError:
            return "Invalid patient ID."
    else:
        pid = int(pid)
    reminders = get_medication_reminders(conn, pid)
    return reminders


def handle_symptom_check(query, conn):
    return simple_symptom_checker(query)


def handle_feedback(query, conn):
    params = extract_parameters_llm(query)
    pid = params.get("patient_id")
    if not pid:
        try:
            pid = int(input("Enter patient ID: "))
        except ValueError:
            return "Invalid patient ID."
    else:
        pid = int(pid)
    try:
        rating = int(input("Rate chatbot response (1-5): "))
        if rating < 1 or rating > 5:
            return "Rating must be between 1 and 5."
    except ValueError:
        return "Invalid rating."
    add_feedback(conn, pid, query, "User feedback placeholder", rating)
    log_intent(query, "feedback", actual_intent="feedback")
    return "Thank you for your feedback."


def handle_chat(query, conn):
    return generate_llm_response(query)


def handle_other(query, conn):
    return "Sorry, I didn't understand your request. Could you please rephrase?"


# ----------- Intent Handler Mapping -----------

INTENT_HANDLERS = {
    "add_patient": handle_add_patient,
    "get_patient": handle_get_patient,
    "update_conditions": handle_update_conditions,
    "update_medications": handle_update_medications,
    "schedule_appointment": handle_schedule_appointment,
    "medication_reminders": handle_medication_reminders,
    "symptom_check": handle_symptom_check,
    "feedback": handle_feedback,
    "chat": handle_chat,
    "other": handle_other,
}

# ----------- Main Chatbot Loop -----------


def main():
    conn = init_db()
    print(
        "Welcome to the Healthcare Chatbot with LLM Intent Detection and API Integration!"
    )
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            intent = detect_intent_llm(user_input)
            handler = INTENT_HANDLERS.get(intent, handle_other)
            response = handler(user_input, conn)
            print(f"[{intent}] {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
