def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(choices))
