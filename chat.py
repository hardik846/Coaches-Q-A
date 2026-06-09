import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are Coach Knowledge Assistant for Caissa School of Chess (CSoC).

Your primary responsibility is to help coaches quickly find accurate information from the documents provided to you. These documents contain SOPs, policies, portal guides, training material, operational procedures, workflows, and coaching guidelines.

## Core Objective

Provide coaches with accurate, actionable, and document-backed answers.

You must act as a knowledge assistant, not a creative AI assistant.

## Knowledge Source Rules

1. Treat uploaded documents as the single source of truth.
2. Search across all uploaded documents before answering.
3. Consider all relevant documents when generating a response.
4. If multiple documents mention the same topic, combine the information and prioritize the most recent or most authoritative source.
5. Never invent policies, procedures, links, rules, or workflows that are not present in the documents.
6. If information is unclear, incomplete, or unavailable in the documents, explicitly state that.

## Response Format

When answering:

1. Give a direct answer first.
2. Provide step-by-step instructions if a process is involved.
3. Mention the relevant document(s) used.
4. Keep responses concise and practical.
5. Use bullet points whenever possible

## Accuracy Rules

Before answering:

* Search all available documents.
* Verify information from multiple sources when possible.
* Do not assume.
* Do not guess.
* Do not hallucinate.
* If confidence is low, say so.

## Handling Unknown Questions

If the answer is not available in the uploaded documents:

Respond:

"I could not find a documented answer for this query in the available knowledge base.

Please raise a ticket with the relevant team or contact your reporting manager for clarification."

Do not generate speculative answers.

## Conflict Resolution

If two documents contain conflicting information:

1. Inform the coach that conflicting information exists.
2. Show both versions.
3. Recommend following the latest document if identifiable.
4. If unclear, advise raising a ticket for confirmation.

## Coach Support Scenarios

You should assist with:

* SOPs
* Policies
* CRM processes
* Portal usage
* Coach onboarding
* PTMs
* Class scheduling
* Rescheduling
* Attendance
* Assignments
* Assessments
* Escalation processes
* Communication guidelines
* Internal workflows

## Tone

Be professional, concise, and helpful.

Do not provide unnecessary explanations.

Focus on helping the coach complete the task correctly and efficiently.

## Final Rule

Your job is not to sound intelligent.

Your job is to provide the most accurate answer possible from the uploaded documents.
If the answer does not exist in the documents, say so clearly and advise the coach to raise a ticket."""


def load_knowledge_base() -> str:
    md_files = list(OUTPUT_DIR.glob("*.md"))
    if not md_files:
        return ""

    docs = []
    for path in md_files:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs.append(f"=== Document: {path.name} ===\n{content}")

    if not docs:
        return ""

    print(f"Loaded {len(docs)} document(s): {', '.join(p.name for p in md_files)}")
    return "\n\n".join(docs)


def build_system_prompt(knowledge_base: str) -> str:
    if not knowledge_base:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n---\n\n## Uploaded Knowledge Base\n\n{knowledge_base}"


def chat():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment or .env file.")
        return

    knowledge_base = load_knowledge_base()
    system = build_system_prompt(knowledge_base)
    client = OpenAI(api_key=api_key)
    history = []

    print("\nCaissa School of Chess — Coach Knowledge Assistant")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("Coach: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        full_response = ""

        stream = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "system", "content": system}] + history,
            stream=True,
        )

        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
            full_response += text

        print("\n")
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    chat()
