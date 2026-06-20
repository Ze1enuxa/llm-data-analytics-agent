import re


DANGEROUS_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"forget\s+previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"api\s*key",
    r"reveal\s+.*key",
    r"show\s+.*key",
    r"\.env",
    r"delete\s+files?",
    r"remove\s+files?",
    r"run\s+shell",
    r"subprocess",
    r"os\.system",
    r"eval\s*\(",
    r"exec\s*\(",
    r"открой\s+\.env",
    r"покажи\s+ключ",
    r"удали\s+файл",
    r"выполни\s+команд",
    r"игнорируй\s+инструкции",
]


def check_user_instruction(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Инструкция не должна быть пустой."

    lowered = text.lower()

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Инструкция отклонена защитой prompt-injection: {pattern}"

    if len(text) > 1500:
        return False, "Инструкция слишком длинная. Сократите запрос до 1500 символов."

    return True, "OK"