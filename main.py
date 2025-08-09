import os, re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = Flask(__name__)

# ---------- CONFIG ----------
def strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

OPENAI_API_KEY = strip_ws(os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")
MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # vision-capable
PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
DEBUG    = os.getenv("DEBUG", "0") == "1"

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- TUTOR PROMPT (micro-lesson + one question; levels only) ----------
MATHMATE_PROMPT = """
MATHMATE — SOCRATIC TUTOR (Acton + Khan) with MICRO-LESSONS

GLOBAL RULES (every turn)
• Teach-while-asking: begin with a tiny MICRO-LESSON (transfer idea, pattern, definition, tip, or common pitfall), then ask ONE question.
• Micro-lesson is short and reusable; it must NOT solve the problem. No question marks in the micro-lesson.
• One-Question Rule: ask EXACTLY ONE question (1 sentence). No lists. No multi-steps.
• Never reveal an operation name and never write an equation. Do NOT say add/subtract/multiply/divide, and do NOT write expressions like 19−5.
• Never give the final answer. Never say correct/incorrect. Use neutral acks (“got it”, “noted”) and move on.
• Keep meta minimal (don’t repeat setup). Teach → ask.
• Style: friendly, concise, 2–3 varied emojis (pool: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔).
• Images: briefly say what you SEE (axes, labels, units, fractions/decimals) in a phrase, then micro-lesson + ONE question.

LEVELS
• Apprentice (precise + defined): use accurate math terms (sum, difference, product, quotient, factor, multiple, numerator/denominator, variable, expression, equation, inequality, rate, slope, intercept, area, perimeter, mean/median/mode, percent). On FIRST use this session, add a 2–6 word parenthesis definition, e.g., “quotient (result of division)”.
• Rising Hero: micro-lesson only if needed (≤1 sentence). Light nudge.
• Master: minimal. No micro-lesson unless asked.

SESSION
• You will receive: level. If level is present, NEVER ask for it again. Start tutoring immediately.

OUTPUT SHAPE
• MICRO-LESSON (0–2 short statements depending on level) → ONE question ending with “?”.
• You may include up to 3 short options like “A) …  B) …  C) …”.
• Absolutely no equations and no operation names.
"""

HARD_CONSTRAINT = (
    "Hard constraint: output a micro-lesson first (0–2 short statements, no '?'), "
    "then EXACTLY ONE question (1 sentence) — total ≤ 3 sentences and only one '?'. "
    "No equations. No operation names."
)

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return "ok", 200

# ---------- UI (white theme, centered title, bubbles; single composer; asks only for level) ----------
@app.get("/")
def home():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>MathMate Pro</title>
<style>
  :root{--bg:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--me:#e6f0ff;--bot:#f8fafc;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial}
  header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:18px 16px;text-align:center}
  header h1{margin:0;font-size:22px;letter-spacing:.2px}
  main{display:flex;justify-content:center}
  .wrap{width:100%;max-width:900px;padding:16px}
  #chat{min-height:58vh;max-height:72vh;overflow:auto;padding:12px 4px}
  .row{display:flex;margin:10px 0}
  .bubble{max-width:72%;padding:12px 14px;border:1px solid var(--line);border-radius:16px;line-height:1.5;white-space:pre-wrap}
  .me{justify-content:flex-end}
  .me .bubble{background:var(--me)}
  .bot{justify-content:flex-start}
  .bot .bubble{background:var(--bot)}
  .sys{color:var(--muted);text-align:center;font-style:italic}
  #panel{position:sticky;bottom:0;background:var(--bg);padding:12px 0;border-top:1px solid var(--line)}
  #unlock{display:flex;gap:8px}
  input,button{font:inherit}
  #password, textarea{padding:12px;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--text)}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--line);background:#111827;color:#fff;cursor:pointer;min-width:84px}
  button:disabled{opacity:.6;cursor:not-allowed}
  #composer{display:none;gap:10px;align-items:flex-end;flex-wrap:wrap}
  #left{flex:1;display:flex;flex-direction:column;gap:8px;min-width:300px}
  textarea{flex:1;resize:vertical;min-height:110px;max-height:300px}
  #drop{border:1px dashed var(--line);border-radius:12px;padding:10px;text-align:center;color:var(--muted)}
  #thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
  .thumb{width:80px;height:80px;border:1px solid var(--line);border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  small.hint{color:var(--muted)}
</style>

<header><h1>🔒 MathMate Pro</h1></header>
<main><div class="wrap">
  <div id="chat"><div class="sys">Type the password to unlock.</div></div>

  <div id="panel">
    <div id="unlock">
      <input id="password" placeholder="Type password…" />
      <button id="unlockBtn">Unlock</button>
    </div>

    <div id="composer">
      <div id="left">
        <textarea id="msg" placeholder="Chat here… (Shift+Enter = newline). You can also paste images."></textarea>
        <div id="drop">
          <label for="fileBtn">➕ Add images (PNG/JPG) — drag & drop or click</label>
          <input id="fileBtn" type="file" accept="image/*" multiple />
          <div id="thumbs"></div>
          <small class="hint">Images are analyzed with your prompt (vision).</small>
        </div>
      </div>
      <button id="sendBtn">Send</button>
    </div>
  </div>
</div></main>

<script>
const chat = document.getElementById('chat');
const unlock = document.getElementById('unlock');
const composer = document.getElementById('composer');
const msgBox = document.getElementById('msg');
const pwdBox = document.getElementById('password');
const unlockBtn = document.getElementById('unlockBtn');
const sendBtn = document.getElementById('sendBtn');
const fileBtn = document.getElementById('fileBtn');
const drop = document.getElementById('drop');
const thumbs = document.getElementById('thumbs');

let AUTH = '';
let LEVEL = '';      // Apprentice | Rising Hero | Master
let queuedImages = [];
let onboarding = true; // ask for level once after unlock

function addBubble(who, text){
  const row = document.createElement('div');
  row.className = who === 'You' ? 'row me' : 'row bot';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.innerHTML = text.replace(/</g,'&lt;');
  row.appendChild(b);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function askLevel(){
  addBubble('MathMate', "Which level should we use—🐣 Apprentice, 🦸 Rising Hero, or 🧠 Master?");
}

function parseLevel(text){
  const t = text.toLowerCase();
  if(/apprentice/.test(t)) return 'Apprentice';
  if(/rising\\s*hero/.test(t)) return 'Rising Hero';
  if(/master/.test(t)) return 'Master';
  return '';
}

async function post(payload){
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json','X-Auth':AUTH},
    body: JSON.stringify({ ...payload, level: LEVEL })
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
drop.addEventListener('dragover', (e)=>{ e.preventDefault(); drop.style.opacity = .9; });
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
  addBubble('You','••••••••');
  const data = await post({ message: pw });
  addBubble('MathMate', data.reply ?? data.error ?? '(error)');
  if(data.reply && data.reply.startsWith('🔓')){
    AUTH = pw;
    unlock.style.display='none';
    composer.style.display='flex';
    onboarding = true;
    askLevel();
    msgBox.focus();
  }
};

sendBtn.onclick = async ()=>{
  let text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;

  // Onboarding: capture level ONCE locally (no model call)
  if(onboarding){
    addBubble('You', text);
    const lvl = parseLevel(text);
    if(lvl){
      LEVEL = lvl;
      onboarding = false;
      addBubble('MathMate', `Great — we’ll use **${LEVEL}** mode. Send your first problem or a photo when you're ready. ✨`);
    }else{
      addBubble('MathMate', "Please choose: Apprentice, Rising Hero, or Master. 🙂");
    }
    msgBox.value = '';
    return;
  }

  // Normal chat
  addBubble('You', text || '(image(s) only)');
  msgBox.value = '';
  sendBtn.disabled = true;
  try{
    const data = await post({ message: text, images: queuedImages });
    addBubble('MathMate', (data.reply ?? data.error ?? '(error)'));
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

# ---------- CHAT (vision + level meta) ----------
@app.post("/chat")
def chat():
    try:
        p = request.get_json(silent=True) or {}
        text   = (p.get("message") or "").strip()
        images = p.get("images") or []
        level  = (p.get("level") or "").strip()

        if not text and not images:
            return jsonify(error="Missing 'message' or 'images'"), 400

        # Auth gate
        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! Let’s pick your level to start.")
            return jsonify(reply="🔒 Please type the access password to begin.")

        # Build user content (vision)
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem."}]

        session_line = (
            f"Session meta: level={level or 'unknown'}. "
            "If level is present, do not ask for it again; start tutoring immediately."
        )

        apprentice_define_rule = ""
        if (level or "").lower() == "apprentice":
            apprentice_define_rule = (
                "Apprentice rule: when you use a precise math term, include a brief 2–6 word "
                "parenthetical definition on its FIRST appearance this session; do not repeat unless asked."
            )
        intensity_line = ""
        if (level or "").lower() == "rising hero":
            intensity_line = "Rising Hero style: give a tiny micro-lesson only if needed; one light nudge."
        elif (level or "").lower() == "master":
            intensity_line = "Master style: minimal; no micro-lesson unless asked; one tiny question."

        messages = [
            {"role": "system", "content": MATHMATE_PROMPT},
            {"role": "system", "content": session_line},
            {"role": "system", "content": apprentice_define_rule},
            {"role": "system", "content": intensity_line},
            {"role": "system", "content": HARD_CONSTRAINT},
            {"role": "user", "content": user_content},
        ]

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=messages,
        )
        return jsonify(reply=completion.choices[0].message.content)

    except Exception as e:
        if DEBUG:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
