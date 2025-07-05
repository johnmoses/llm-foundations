from db import init_db
from users.users import add_user
from llm.prompts_llm import LLMWrapper
from conversational.dispatcher import handle_function_call


def main():
    init_db()
    username = input("Enter your username: ").strip()
    user_id = add_user(username)

    llm = LLMWrapper(model_path="/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf")

    print(f"Welcome {username}! Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Get function call JSON from LLM
        response_json = llm.function_calling(user_input)
        print(f"Debug: LLM response JSON:\n{response_json}\n")  # Optional debug output

        # Dispatch function call and get user-friendly reply
        reply = handle_function_call(response_json, user_id=user_id)
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()
