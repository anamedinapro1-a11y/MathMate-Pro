import os, re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = Flask(__name__)

# -------------------- CONFIG --------------------
def clean_key(k: str) -> str:
    return re.sub(r"\s+", "", (k or ""))

OPENAI_API_KEY = clean_key(os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # vision-capable
PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
DEBUG    = os.getenv("DEBUG", "0") == "1"

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- TUTOR PROMPT (ONE-STEP SOCratic) --------------------
MATHMATE_PROMPT = """
MATHMATE — ONE-STEP SOCRATIC TUTOR (Acton + Khan)

GLOBAL RULES (must follow every turn)
• One-Question Rule: ask EXACTLY ONE short question (≤2 sentences). No lists. No multi-steps.
• Never reveal the operation or write an equation. Do NOT say “add/subtract/multiply/divide”, and do NOT show expressions like 19−5. Let the learner decide.
• Never give the final answer. Never say correct/incorrect. Use neutral acks (“got it”, “noted”) and move on.
• Tone: friendly, concise, 2–3 emojis max, vary emojis across turns (rotate from: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔).
• Images: briefly describe what you SEE (axes, labels, units, fractions/decimals) without solving, then ask ONE clarifying question.

LEVELS
• Apprentice (simple words): avoid terms like “quantities/difference/compute”. Prefer kid-friendly words (“numbers”, “how many”, “total”). If a tricky word is needed, explain it quickly in (parentheses).
• Rising Hero: slightly bigger steps, still one question.
• Master: be minimal. Ask the smallest helpful question; one short sentence if possible.

FLOW / PLANNING
• If the user has NOT told you their level and total # of questions:
  – First, ask for the level: “Apprentice / Rising Hero / Master?”
  – Then, ask: “How many total questions are in this exercise?”
• After the level AND total are known:
  – If level is Apprentice or Rising Hero: briefly announce the plan ONCE: “I’ll guide ~40%, you’ll teach back ~50%, last 10% I’ll just be here for questions.” Keep this to one sentence with an emoji, then continue with ONE question.
  – If level is Master: just say “Okay.” and continue with one minimal question.
• Always keep replies ONE question at a time. No bullet lists.

OUTPUT SHAPE
• Begin with a tiny nudge + ONE question ending with “?” Include 2–3 varied emojis. No equations, no operation names.
"""

HARD_CONSTRAINT = (
    "Hard constraint: reply with ONE short question only (<=2 sentences), "
    "no equations, no operation names, end with a single '?' and nothing after."
)

# -------------------- HEALTH --------------------
@app.get("/health")
def health():
    return "ok", 200

# -------------------- UI --------------------
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
  #chat{flex:1;min-height:60vh;max-height:72vh;overflow:auto;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
  .row{margin:10px 0;line-height:1.5;white-space:pre-wrap}
  .me b{color:#93c5fd}
  .bot b{color:#86efac}
  .sys{color:var(--muted);font-style:italic}
  #panel{position:sticky;bottom:0;max-width:1000px;margin:12px auto 28px;display:flex;flex-direction:column;gap:10px;padding:0 16px}
  #pwdWrap{display:flex;gap:8px}
  #password{flex:1;padding:12px;border-radius:12px;border:1px solid var(--border);background:#0f172a;color:var(--text)}
  #composer{display:none;gap:10px;align-items:flex-end}
  #left{flex:1;display:flex;flex-direction:column;gap:8px}
  textarea{flex:1;resize:vertical;min-height:110px;max-height:300px;padding:12px;border-radius:12px;border:1px solid var(--border);background:#0f172a;color:var(--text)}
  #drop{border:1px dashed var(--border);border-radius:12px;padding:10px;text-align:center;color:var(--muted)}
  #thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
  .thumb{width:80px;height:80px;border:1px solid var(--border);border-radius:8px;background:#0f172a;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--border);background:#111827;color:var(--text);cursor:pointer;min-width:80px}
  button:disabled{opacity:.6;cursor:not-allowed}
  input[type=file]{display:none}
  small.hint{color:var(--muted)}
</style>

<header>🔒 MathMate Pro</header>
<main>
  <div id="chat"><div class="row sys">Type the password to unlock.</div></div>
</main>

<div id="panel">
  <div id="pwdWrap">
    <input id="password" placeholder="Type password…" />
    <button id="unlockBtn">Unlock</button>
  </div>

  <div id="composer">
    <div id="left">
      <textarea id="msg" placeholder="Ask MathMate… (Shift+Enter = newline)"></textarea>
      <div id="drop">
        <label for="fileBtn">➕ Add images (PNG/JPG) — drag & drop or click</label>
        <input id="fileBtn" type="file" accept="image/*" multiple />
        <div id="thumbs"></div>
        <small class="hint">Images will be analyzed with the prompt (vision).</small>
      </div>
    </div>
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
const fileBtn = document.getElementById('fileBtn');
const drop = document.getElementById('drop');
const thumbs = document.getElementById('thumbs');

let AUTH = '';
let queuedImages = [];

function addRow(who, text){
  const div = document.createElement('div');
  div.className = 'row ' + (who==='You'?'me':'bot');
  div.innerHTML = `<b>${who}:</b> ${text.replace(/</g,'&lt;')}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function post(payload){
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json','X-Auth':AUTH},
    body: JSON.stringify(payload)
  });
  return r.json();
}

function addThumb(src){
  const d = document.createElement('div');
  d.className = 'thumb';
  const img = document.createElement('img');
  img.src = src;
  d.appendChild(img);
  thumbs.appendChild(d);
}

function fileToDataURL(file){
  return new Promise((res, rej)=>{
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = rej;
    fr.readAsDataURL(file);
  });
}

fileBtn.onchange = async (e)=>{
  for(const f of e.target.files){
    const dataURL = await fileToDataURL(f);
    queuedImages.push(dataURL);
    addThumb(dataURL);
  }
  fileBtn.value = '';
};

drop.addEventListener('dragover', (e)=>{ e.preventDefault(); drop.style.opacity = .8; });
drop.addEventListener('dragleave', ()=>{ drop.style.opacity = 1; });
drop.addEventListener('drop', async (e)=>{
  e.preventDefault(); drop.style.opacity = 1;
  const files = Array.from(e.dataTransfer.files || []);
  for(const f of files){
    if(!f.type.startsWith('image/')) continue;
    const dataURL = await fileToDataURL(f);
    queuedImages.push(dataURL);
    addThumb(dataURL);
  }
});

unlockBtn.onclick = async ()=>{
  const pw = (pwdBox.value||'').trim();
  if(!pw) return;
  addRow('You','••••••••');
  const data = await post({ message: pw });
  addRow('MathMate', data.reply ?? data.error ?? '(error)');
  if(data.reply && data.reply.startsWith('🔓')){
    AUTH = pw;
    pwdWrap.style.display='none';
    composer.style.display='flex';
    msgBox.focus();
  }
};

sendBtn.onclick = async ()=>{
  const text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;
  addRow('You', text || '(image(s) only)');
  msgBox.value = '';
  sendBtn.disabled = true;
  try{
    const data = await post({ message: text, images: queuedImages });
    addRow('MathMate', (data.reply ?? data.error ?? '(error)'));
  }finally{
    sendBtn.disabled = false;
    queuedImages = [];
    thumbs.innerHTML = '';
    msgBox.focus();
  }
};

msgBox.addEventListener('keydown', (e)=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendBtn.click(); }
});
pwdBox.addEventListener('keydown', (e)=>{
  if(e.key==='Enter'){ e.preventDefault(); unlockBtn.click(); }
});
</script>
"""

# -------------------- CHAT (vision + strict style) --------------------
@app.post("/chat")
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("message") or "").strip()
        images = payload.get("images") or []

        if not text and not images:
            return jsonify(error="Missing 'message' or 'images'"), 400

        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! How many total questions are in this exercise, and which level: 🐣 Apprentice / 🦸 Rising Hero / 🧠 Master?")
            return jsonify(reply="🔒 Please type the access password to begin.")

        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem."}]

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": MATHMATE_PROMPT},
                {"role": "system", "content": HARD_CONSTRAINT},
                {"role": "user", "content": user_content},
            ],
        )
        return jsonify(reply=completion.choices[0].message.content)

    except Exception as e:
        if DEBUG:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# -------------------- LOCAL RUN --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
