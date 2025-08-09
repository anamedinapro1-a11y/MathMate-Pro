import os
from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")  # required for session cookies

# --- OpenAI setup ---
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
DEBUG = os.getenv("DEBUG", "0") == "1"

client = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/health")
def health():
    return "ok", 200

# --- tutor prompt ---
MATHMATE_PROMPT = """
🎯 MATHMATE – ACTON + KHAN ACADEMY AI GUIDE (COMPRESSED)
- Socratic guide only: ask questions/options; never confirm correctness; never give final answers.
- Levels: Apprentice (slow, define terms, step-by-step), Rising Hero (short nudge), Master (student leads).
- Khan image rules: detect fraction/decimal; don’t reveal coordinates; ask x vs y; format awareness.
- Quiz flow: ask total # of questions; plan 40% guide / 50% teach-back / 10% hands-off; announce each question.
- Tone: respectful, encouraging, concise unless Apprentice is chosen.
"""

# --- UI page ---
@app.get("/")
def home():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>MathMate Pro</title>
<style>
  :root{--card:#111827;--text:#e5e7eb;--muted:#9ca3af;--border:#374151}
  *{box-sizing:border-box}
  body{margin:0;background:#0b1220;color:var(--text);font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial}
  header{position:sticky;top:0;background:#0b1220;border-bottom:1px solid var(--border);padding:14px 18px;font-weight:700}
  main{display:flex;gap:16px;max-width:1000px;margin:0 auto;padding:16px}
  #chat{flex:1;min-height:60vh;max-height:70vh;overflow:auto;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
  .row{margin:10px 0;line-height:1.45}
  .me b{color:#93c5fd}
  .bot b{color:#86efac}
  .sys{color:var(--muted);font-style:italic}
  #panel{position:sticky;bottom:0;max-width:1000px;margin:12px auto 28px;display:flex;gap:10px;padding:0 16px}
  #pwdWrap{flex:1;display:flex;gap:8px}
  #password{flex:1;padding:12px;border-radius:12px;border:1px solid var(--border);background:#0f172a;color:var(--text)}
  #composer{display:none;flex:1;gap:8px}
  textarea{flex:1;resize:vertical;min-height:90px;max-height:240px;padding:12px;border-radius:12px;border:1px solid var(--border);background:#0f172a;color:var(--text)}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--border);background:#111827;color:var(--text);cursor:pointer}
  button:disabled{opacity:.6;cursor:not-allowed}
</style>
<header>🔒 MathMate Pro</header>
<main>
  <div id="chat">
    <div class="row sys">Type the password to unlock.</div>
  </div>
</main>

<div id="panel">
  <div id="pwdWrap">
    <input id="password" placeholder="Type password…" />
    <button id="unlockBtn">Unlock</button>
  </div>

  <div id="composer">
    <textarea id="msg" placeholder="Ask MathMate… (Shift+Enter = newline)"></textarea>
    <button id="sendBtn">Send</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const pwdWrap = document.getElementById('pwdWrap');
const composer = document.getElementById('composer');
const msgBox = document.getElementById('msg');
const pwdBox = document.getElementById('password');
const unlockBtn = document.getElementById('unlockBtn');
const sendBtn = document.getElementById('sendBtn');

function addRow(who, text){
  const div = document.createElement('div');
  div.className = 'row ' + (who==='You'?'me':'bot');
  div.innerHTML = `<b>${who}:</b> ${text.replace(/</g,'&lt;')}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function post(message){
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    credentials:'include',
    body: JSON.stringify({ message })
  });
  return r.json();
}

unlockBtn.onclick = async () => {
  const pw = (pwdBox.value||'').trim();
  if(!pw) return;
  addRow('You', '••••••••');
  const data = await post(pw);
  addRow('MathMate', data.reply ?? data.error ?? '(error)');
  if(data.reply && data.reply.startsWith('🔓')){
    pwdWrap.style.display = 'none';
    composer.style.display = 'flex';
    msgBox.focus();
  }
};

sendBtn.onclick = async () => {
  const m = (msgBox.value||'').trim();
  if(!m) return;
  addRow('You', m);
  msgBox.value = '';
  sendBtn.disabled = true;
  try{
    const data = await post(m);
    addRow('MathMate', (data.reply ?? data.error ?? '(error)'));
  }finally{
    sendBtn.disabled = false;
    msgBox.focus();
  }
};

msgBox?.addEventListener('keydown', (e)=>{
  if(e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    sendBtn.click();
  }
});
pwdBox?.addEventListener('keydown', (e)=>{
  if(e.key==='Enter'){ e.preventDefault(); unlockBtn.click(); }
});
</script>
"""

# --- Chat route ---
@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()

        if not msg:
            return jsonify(error="Missing 'message'"), 400

        # password gate
        if not session.get("unlocked"):
            if msg.lower() == PASSWORD.lower():
                session["unlocked"] = True
                return jsonify(reply="🔓 Unlocked! How many total questions are in this exercise, and which level: 🐣 Apprentice / 🦸 Rising Hero / 🧠 Master?")
            return jsonify(reply="🔒 Please type the access password to begin.")

        # tutoring
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

    except Exception as e:
        app.logger.exception("Chat route crashed")
        if DEBUG:
            return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# --- local run ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
