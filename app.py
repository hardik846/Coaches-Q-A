import os
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

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
3. Keep responses concise and practical.
4. Use bullet points whenever possible

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

## Tone

Be professional, concise, and helpful.

Do not provide unnecessary explanations.

Focus on helping the coach complete the task correctly and efficiently.

## Final Rule

Your job is not to sound intelligent.

Your job is to provide the most accurate answer possible from the uploaded documents.
If the answer does not exist in the documents, say so clearly and advise the coach to raise a ticket."""


def load_knowledge_base() -> str:
    if not OUTPUT_DIR.exists():
        return ""
    md_files = sorted(OUTPUT_DIR.glob("*.md"))
    if not md_files:
        return ""
    docs = []
    for path in md_files:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs.append(f"=== Document: {path.name} ===\n{content}")
    return "\n\n".join(docs)


KNOWLEDGE_BASE = load_knowledge_base()
FULL_SYSTEM_PROMPT = (
    f"{SYSTEM_PROMPT}\n\n---\n\n## Uploaded Knowledge Base\n\n{KNOWLEDGE_BASE}"
    if KNOWLEDGE_BASE
    else SYSTEM_PROMPT
)

def log_to_sheet(name: str, question: str, answer: str):
    try:
        script_url = os.environ.get("GOOGLE_SCRIPT_URL")
        if not script_url:
            return
        payload = json.dumps({
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "question": question,
            "answer": answer,
        }).encode("utf-8")
        req = urllib.request.Request(
            script_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # logging failure must never break the chat


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing message"}), 400

        history = data.get("history", [])
        user_message = data["message"].strip()
        coach_name = data.get("name", "Unknown").strip() or "Unknown"
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return jsonify({"error": "OPENAI_API_KEY not configured on server"}), 500

        client = OpenAI(api_key=api_key)

        messages = [{"role": "system", "content": FULL_SYSTEM_PROMPT}]
        messages += history
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=messages,
        )

        reply = response.choices[0].message.content
        log_to_sheet(coach_name, user_message, reply)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
