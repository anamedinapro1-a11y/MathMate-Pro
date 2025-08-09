import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------- App & Config --------------------
app = Flask(__name__)

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # vision-capable
PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
DEBUG = os.getenv("DEBUG", "0") == "1"

client = OpenAI(api_key=OPENAI_API_KEY)

MATHMATE_PROMPT = """
🎯 MATHMATE – ACTON + KHAN ACADEMY AI GUIDE (COMPRESSED)
- Socratic guide only: ask questions/options; never confirm correctness; never give final answers.
- Levels: Apprentice (slow, define terms, step-by-step), Rising Hero (short nudge), Master (student leads).
- For image problems: first describe what you see (axes, labels, units, fractions/decimals), then ask 1 clarifying question before proceeding.
- Quiz flow: ask total # of questions; plan 40% guide / 50% teach-back / 10% hands-off; announce each question.
- Tone: respectful, encouraging, concise unless Apprentice is chosen.
"""

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
let queuedImages = []; // data URLs

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
  const wrapper = document.createElement('div');
  wrapper.className = 'thumb';
  const img = document.createElement('img');
  img.src = src;
  wrapper.appendChild(img);
  thumbs.appendChild(wrapper);
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
  const data = await post({ message: pw });   // try password
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
    // clear images after sending
    queuedImages = [];
    thumbs.innerHTML = '';
    msgBox.focus();
  }
};

// enter to send; shift+enter newline
msgBox.addEventListener('keydown', (e)=>{
  if(e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    sendBtn.click();
  }
});
pwdBox.addEventListener('keydown', (e)=>{
  if(e.key==='Enter'){ e.preventDefault(); unlockBtn.click(); }
});
</script>
"""

# -------------------- Chat (supports images) --------------------
@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("message") or "").strip()
        images = data.get("images") or []  # list of data URLs (e.g., "data:image/png;base64,...")

        if not text and not images:
            return jsonify(error="Missing 'message' or 'images'"), 400

        auth = request.headers.get("X-Auth", "")

        # Unlock flow: if not authorized, only accept the password as the text message
        if auth != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! How many total questions are in this exercise, and which level: 🐣 Apprentice / 🦸 Rising Hero / 🧠 Master?")
            return jsonify(reply="🔒 Please type the access password to begin.")

        # Build a vision-aware message
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            # Send the data URL directly; OpenAI supports "image_url" with base64 data URLs
            user_content.append({"type": "image_url", "image_url": {"url": url}})

        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem step-by-step."}]

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": MATHMATE_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        reply = completion.choices[0].message.content
        return jsonify(reply=reply)

    except Exception as e:
        if DEBUG:
            return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# -------------------- Local run --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
