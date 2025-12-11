# cli.py
from agents.unified_agent import get_music_agent

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
            resp = agent.run(q)
            print("\nAgent:\n", resp, "\n")
        except Exception as e:
            print("\n[Error]", e, "\n")

if __name__ == "__main__":
    main()

