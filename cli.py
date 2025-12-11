from typing import Any, Iterable
from agents.unified_agent import get_music_agent


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or part.get("data")
                if text:
                    parts.append(str(text))
            elif part:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _format_agent_response(resp: Any) -> str:
    """LangGraph agents return dicts with messages; normalize to readable text."""
    if isinstance(resp, dict):
        for key in ("output", "result"):
            value = resp.get(key)
            if value:
                return str(value)
        messages = resp.get("messages")
        if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes)):
            try:
                last_message = list(messages)[-1]
            except IndexError:
                last_message = None
            if last_message is not None:
                content = getattr(last_message, "content", last_message)
                return _stringify_message_content(content)
        return str(resp)
    return str(resp)


def main():
    agent = get_music_agent()
    print("🎵 AI Music Copilot (RAG Mode)")
    print("Ask about artists, albums, moods, recommendations.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if q.strip().lower() in {"exit", "quit"}:
            break

        if not q.strip():
            continue  # ignore empty input

        try:
            resp = agent.invoke({"input": q})
            answer = _format_agent_response(resp).strip()
            print("\nAgent:\n", answer or "[No response]", "\n")
        except Exception as e:
            print("\n[Error]", e, "\n")


if __name__ == "__main__":
    main()
