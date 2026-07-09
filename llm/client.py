"""
Provider-agnostic LLM interface.

One place the vendor lives. Everything else (anomaly alerts, the NL query layer)
calls chat() / embed() / available() and never imports a vendor SDK directly, so
switching providers is an env change, not a code change.

    LLM_PROVIDER   openai (default) | anthropic
    LLM_MODEL      chat/completions model (default per provider)
    EMBED_MODEL    embeddings model (OpenAI only; Anthropic has no embeddings API)

Provider SDKs are imported lazily so the unused one need not be installed.
"""
import os

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

_DEFAULT_CHAT = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-6"}
CHAT_MODEL = os.getenv("LLM_MODEL", _DEFAULT_CHAT.get(PROVIDER, "gpt-4o"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small = 1536

_PLACEHOLDERS = {"", "your_key_here", "your-key-here"}


def available() -> bool:
    """True if a real key is configured for the active provider."""
    key = os.getenv("OPENAI_API_KEY" if PROVIDER == "openai" else "ANTHROPIC_API_KEY", "")
    return key not in _PLACEHOLDERS


def chat(system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
    """Single-turn completion. Returns the assistant's text."""
    if PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=CHAT_MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. OpenAI only (Anthropic has no embeddings API)."""
    if PROVIDER != "openai":
        raise NotImplementedError(
            f"Embeddings require an OpenAI-compatible provider; LLM_PROVIDER={PROVIDER}. "
            "Set LLM_PROVIDER=openai (or point EMBED at an OpenAI-compatible endpoint)."
        )
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]
