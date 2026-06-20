import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from gigachat import GigaChat


def get_secret(name: str, default: str | None = None) -> str | None:
    load_dotenv()

    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return default


def get_verify_ssl_flag() -> bool:
    raw_value = get_secret("GIGACHAT_VERIFY_SSL_CERTS", "false")
    return str(raw_value).strip().lower() not in {"false", "0", "no", "off"}


def create_gigachat_client() -> GigaChat:
    auth_key = get_secret("GIGACHAT_AUTH_KEY")

    if not auth_key:
        raise RuntimeError(
            "GIGACHAT_AUTH_KEY is not set. Add it to .env or Streamlit Secrets."
        )

    return GigaChat(
        credentials=auth_key,
        verify_ssl_certs=get_verify_ssl_flag(),
    )


def extract_text_from_response(response: Any) -> str:
    if hasattr(response, "choices"):
        content = response.choices[0].message.content
        return content if isinstance(content, str) else str(content)

    if hasattr(response, "text"):
        return response.text

    return str(response)


def ask_gigachat(prompt: str) -> str:
    with create_gigachat_client() as client:
        response = client.chat(prompt)

    return extract_text_from_response(response)