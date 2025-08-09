import os
from flask import Flask, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me-please")

app = Flask(__name__)
app.secret_key = SESSION_SECRET
client = OpenAI(api_key=API_KEY)

MATHMATE_PROMPT = """
🎯 MATHMATE – ACTON + KHAN ACADEMY AI GUIDE (COMPRESSED)
- Socratic guide only: ask questions/options; never confirm correctness; never give final answers.
- Levels: Apprentice (slow, define terms, step-by-step), Rising Hero (short nudge), Master (student leads; “What’s your first step?”).
- Khan image rules: detect fraction/decimal; don’t reveal coordinates; ask x vs y; if format mismatch, nudge to match.
- Quiz flow: ask total # of questions; plan 40% guide / 50% teach-back / 10% hands-off; ask to announce each new question.
- Tone: respectful, encouraging, curious; never condescending.
- Accuracy: use precise calculations; be format-aware; concise unless Apprentice is chosen.
"""

@app.route("/", methods=["GET"])
def home():
    # simple web UI so the root URL works
    return """
<!doctype html>
<meta charset="utf-8" />
<title>MathMate Pro</title>
<style>
  body{font-family:system-ui;margin:24px;max-width:800px}
  #chat{border:1px solid #ddd;padding:12px;height:380px;overflow:auto;border-radius:12px}
  input,button{padding:10px;margin-top:10px}
  button{cursor:pointer}
</style>
<h2>🔒 MathMate Pro</h2>
<div id="chat"></div>
<input id="msg" placeholder="Type password first…"/>
<button onclick="send()">Send</button>
<script>
async function send(){
  const box = document.getElementById('msg');
  const chat = document.getElementById('chat');
  const m = box.value.trim(); if(!m) return;
  chat.innerHTML += `<p><b>You:</b> ${m}</p>`;
  box.value='';
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({message:m}),
    credentials:'include'
  });
  const data = await r.json();
  chat.innerHTML += `<p><b>MathMate:</b> ${data.reply}</p>`;
  chat.scrollTop = chat.scrollHeight;
}
</script>
"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = (data.get("message") or "").strip()

    # password gate
    if not session.get("unlocked"):
        if msg.lower() == PASSWORD.lower():
            session["unlocked"] = True
            return jsonify(reply="🔓 Unlocked! How many total questions are in this exercise, and which level: 🐣 Apprentice / 🦸 Rising Hero / 🧠 Master?")
        return jsonify(reply="🔒 Please type the access password to begin."), 401

    # normal tutoring
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role":"system","content":MATHMATE_PROMPT},
            {"role":"user","content":msg}
        ],
    )
    return jsonify(reply=completion.choices[0].message.content)

if __name__ == "__main__":
    # Replit proxy will forward this automatically
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
