import os
from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY)


PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# OpenAI SDK reads OPENAI_API_KEY from env automatically
client = OpenAI()

@app.get("/health")
def health():
    return "ok", 200

# --- tutor prompt (short) ---
MATHMATE_PROMPT = """
🎯 MATHMATE – ACTON + KHAN ACADEMY AI GUIDE (COMPRESSED)
- Socratic guide only: ask questions/options; never confirm correctness; never give final answers.
- Levels: Apprentice (slow, define terms, step-by-step), Rising Hero (short nudge), Master (student leads).
- Khan image rules: detect fraction/decimal; don’t reveal coordinates; ask x vs y; format awareness.
- Quiz flow: ask total # of questions; plan 40% guide / 50% teach-back / 10% hands-off; announce each question.
- Tone: respectful, encouraging, concise unless Apprentice is chosen.
"""

# --- very simple UI so your root URL works ---
@app.get("/")
def home():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>MathMate Pro</title>
<style>
  body{font-family:system-ui;margin:24px;max-width:820px}
  #chat{border:1px solid #ddd;padding:12px;height:420px;overflow:auto;border-radius:12px}
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
  try{
    const r = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: m }),   // <— backend expects "message"
      credentials:'include'
    });
    const data = await r.json();
    chat.innerHTML += `<p><b>MathMate:</b> ${data.reply ?? data.error}</p>`;
  }catch(e){
    chat.innerHTML += `<p><b>MathMate:</b> (network error)</p>`;
  }
  chat.scrollTop = chat.scrollHeight;
}
</script>
"""

# --- single, clean /chat route ---
@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()  # matches frontend key

        if not msg:
            return jsonify(error="Missing 'message'"), 400

        # 🔒 Password gate first
        if not session.get("unlocked"):
            if msg.lower() == PASSWORD.lower():
                session["unlocked"] = True
                return jsonify(reply="🔓 Unlocked! How many total questions are in this exercise, and which level: 🐣 Apprentice / 🦸 Rising Hero / 🧠 Master?")
            return jsonify(reply="🔒 Please type the access password to begin.")

        # 🤖 Normal tutoring
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": MATHMATE_PROMPT},
                {"role": "user", "content": msg},
            ],
        )
        reply = completion.choices[0].message.content
        return jsonify(reply=reply)

    except Exception:
        app.logger.exception("Chat route crashed")
        return jsonify(error="Server error"), 500


# --- local run (Railway/Gunicorn will ignore this) ---
if __name__ == "__main__":
    # Railway sets PORT; default to 8080 which we exposed
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
