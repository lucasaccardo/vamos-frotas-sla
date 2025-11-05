import os
import time
import requests
from typing import Iterable, List, Dict, Any, Generator

# Histórico do app: [{"role": "user"|"assistant"|"model", "parts": [{"text": "..."}]}]

def _persona_ptbr() -> str:
    return (
        "Você é o Assistente I.A. do app Frotas Vamos SLA. "
        "Fale em português do Brasil, de forma natural, humana e objetiva. "
        "Evite jargões desnecessários, ofereça opções quando fizer sentido, "
        "faça perguntas de esclarecimento se faltar contexto e mantenha um tom cordial. "
        "Responda de forma clara e prática, sem parecer robótico."
    )

def detect_provider() -> str:
    try:
        import streamlit as st
        return str(st.secrets.get("AI_PROVIDER", "gemini")).strip().lower()
    except Exception:
        return (os.getenv("AI_PROVIDER") or "gemini").strip().lower()

def get_model_name(provider: str) -> str:
    # Permite override via Secret/Env
    override = ""
    try:
        import streamlit as st
        override = str(st.secrets.get("MODEL_NAME", st.secrets.get("MODEL_OVERRIDE", ""))).strip()
    except Exception:
        override = (os.getenv("MODEL_NAME") or os.getenv("MODEL_OVERRIDE") or "").strip()
    if override:
        return override

    if provider == "huggingface":
        # Modelo sugerido (bom e gratuito no HF Inference API).
        # Você pode trocar por "google/gemma-2-9b-it" ou "meta-llama/Meta-Llama-3-8B-Instruct"
        # (para alguns, é preciso aceitar a licença na página do modelo no HF).
        return "Qwen/Qwen2.5-7B-Instruct"
    # Defaults de outros provedores (se você ativar futuramente)
    if provider == "openai":
        return "gpt-4o-mini"
    if provider == "anthropic":
        return "claude-3-5-haiku-latest"
    if provider == "openrouter":
        return "google/gemini-2.0-flash-lite"
    return "gemini-2.5-flash"

def _convert_history_for_chat(history: List[Dict[str, Any]], max_turns: int = 6) -> str:
    """Concatena sistema + histórico em um prompt estilo 'instruções' + diálogo."""
    system = _persona_ptbr()
    lines = [f"### Instruções (pt-BR)\n{system}\n"]
    if history:
        lines.append("### Conversa (histórico)\n")
        for m in history[-max_turns:]:
            role = "Assistente" if m.get("role") in ("assistant", "model") else "Usuário"
            text = (m.get("parts") or [{}])[0].get("text") or ""
            lines.append(f"{role}: {text}")
    lines.append("\n### Tarefa\nResponda em português (Brasil), de forma natural e direta.")
    return "\n".join(lines).strip()

def _simulate_stream(text: str, chunk_chars: int = 60, delay: float = 0.008) -> Generator[str, None, None]:
    """Fatia o texto em blocos pequenos para simular streaming no UI."""
    if not text:
        return
    for i in range(0, len(text), chunk_chars):
        yield text[i : i + chunk_chars]
        if delay:
            time.sleep(delay)

def get_ai_stream(
    provider: str,
    prompt: str,
    temperature: float,
    history: List[Dict[str, Any]],
) -> Iterable[str]:
    """
    Retorna um iterável de strings (stream de fragmentos).
    Implementa Hugging Face Inference API quando provider == 'huggingface'.
    """
    if provider != "huggingface":
        raise RuntimeError(f"AI_PROVIDER '{provider}' não suportado nesta instalação (use 'huggingface').")

    model = get_model_name(provider)
    # Pega a chave do HF dos Secrets ou Env
    try:
        import streamlit as st
        hf_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
    except Exception:
        hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        raise RuntimeError("Defina HUGGINGFACE_API_KEY nos Secrets para usar o provedor 'huggingface'.")

    # Monta prompt com persona + histórico + turno atual
    dialog = _convert_history_for_chat(history)
    full_prompt = f"{dialog}\n\nUsuário: {prompt}\nAssistente:"

    # Parâmetros do Inference API (text-generation)
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": float(temperature),
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_key}"}

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code == 403:
        raise RuntimeError(
            f"Acesso negado ao modelo '{model}'. Entre no Hugging Face e aceite a licença na página do modelo."
        )
    if resp.status_code == 404:
        raise RuntimeError(f"Modelo '{model}' não encontrado no Inference API.")
    if resp.status_code in (429, 503):
        raise RuntimeError(
            f"Modelo ocupado/limitado agora (status {resp.status_code}). Tente novamente em alguns segundos."
        )
    if not resp.ok:
        raise RuntimeError(f"Falha no Inference API ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Inference API retornou erro: {data['error']}")
    if isinstance(data, list) and data and "generated_text" in data[0]:
        text = data[0]["generated_text"]
    else:
        text = str(data)

    # Simula streaming para a UI
    return _simulate_stream(text)
