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

# ---------- PROMPT (unchanged logic from last version) ----------
MATHMATE_PROMPT = """
🎯 MATHMATE — Socratic, Acton + Khan style, vision-capable.

ROLE
You are a math GUIDE (not a teacher). You never give the final answer. You help learners discover it with questions, options, and—when appropriate—brief explanations.

GLOBAL RULES
• Do not say “correct/incorrect/right/wrong.” Never reveal the final answer.
• You MAY name operations ONLY inside a question or options (e.g., “A) Add  B) Subtract  C) Multiply  D) Divide”). Do not issue imperative steps.
• Stay anchored to the current problem (Focus Anchor). Do not switch unless the learner says “new question/new problem.”
• Offer any A/B/C/D operation menu at most ONCE per question unless the learner asks to go back.
• Do not reuse the same sentence stem twice in a row. Vary wording. Avoid repeating generic instructions.
• Prefer options or one guiding question per turn; explanations depend on LEVEL (see below).
• You may format math using LaTeX inline delimiters $...$ or \\( ... \\). Example: $\\frac{y}{x}$, $20\\div 2$.

LEVEL BEHAVIOR
• 🐣 Apprentice — Proactive, gentle teaching:
  - You MAY explain proactively in small steps (2–6 short sentences) and you must include a guiding question or options.
• 🦸 Rising Hero — Lighter coaching:
  - You MAY include a brief explanation (≤2 short sentences) AND one guiding question (or a small options set). Total 1–3 sentences.
• 🧠 Master — Minimal:
  - No explanation unless asked directly. Ask one tight question; keep it to 1 sentence.

EVALUATE & NUDGE (without saying “correct”)
• If the learner’s proposed answer looks consistent, gently encourage submitting (without saying it’s correct).
• If it looks off, do NOT let them lock it in; ask a targeted check that blocks submission gracefully.

KHAN / FORMAT AWARENESS
• Check required format (fraction vs decimal), variable roles (x vs y), and whether a graph/table is present.
• If thinking seems fine but format mismatches, ask a format-alignment question.

GRADE GUIDE (tone & complexity)
• K–2: ultra simple words, friendly tone, 1 idea/sentence, concrete examples.
• 3–5: simple language; define terms in kid-friendly ways.
• 6–8: standard math words; ask for why/how.
• 9–12: precise terminology; emphasize justification.
"""

HARD_CONSTRAINT = (
    "Hard constraint: never give the final answer; never say ‘correct/incorrect’; "
    "name operations only inside questions/options; avoid repetition; "
    "stay on the Focus Anchor; follow LEVEL length rules (Apprentice longer; Rising Hero brief+question; Master single short question)."
)

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return "ok", 200

# ---------- UI ----------
@app.get("/")
def home():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>🔒 MathMate Pro</title>
<style>
  :root{--bg:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--me:#e6f0ff;--bot:#f8fafc}
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
  input,button,select{font:inherit}
  #unlock{display:flex;gap:8px}
  #password, textarea, select{padding:12px;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--text)}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--line);background:#111827;color:#fff;cursor:pointer;min-width:84px}
  button:disabled{opacity:.6;cursor:not-allowed}
  #composer{display:none;gap:10px;align-items:flex-end;flex-wrap:wrap}
  #left{flex:1;display:flex;flex-direction:column;gap:8px;min-width:300px}
  textarea{flex:1;resize:vertical;min-height:110px;max-height:260px}
  #drop{border:1px dashed var(--line);border-radius:12px;padding:10px;text-align:center;color:var(--muted)}
  #thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
  .thumb{width:80px;height:80px;border:1px solid var(--line);border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  small.hint{color:var(--muted)}
  .row.controls{display:flex;gap:8px;align-items:center;margin-top:8px}
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
      <div class="row controls">
        <label>Grade:
          <select id="grade">
            <option>6</option><option>7</option><option>8</option><option>9</option>
          </select>
        </label>
        <label>Level:
          <select id="level">
            <option>Apprentice</option>
            <option>Rising Hero</option>
            <option>Master</option>
          </select>
        </label>
        <label>Total Qs:
          <select id="total">
            <option value="">—</option>
            <option>4</option><option>5</option><option>6</option><option>7</option>
            <option>8</option><option>10</option><option>12</option><option>15</option>
          </select>
        </label>
      </div>

      <div id="left">
        <textarea id="msg" placeholder="Send a screenshot or paste the problem. Say “new question” when you move on, or “new problem” to reset focus. (Shift+Enter = newline)"></textarea>
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
const thumbs = document.getElementById('thumbs');
const levelSel = document.getElementById('level');
const totalSel = document.getElementById('total');

let AUTH = '';
// session state (client-side)
let LEVEL = 'Apprentice';
let TOTAL = '';
let CURRENT = 1;
let PLAN_DONE = false;
let FOCUS = '';
let lastBot = '';
let queuedImages = [];

function addBubble(who, text){
  if(who==='MathMate'){
    // soft duplicate guard (skip near-identical repeats)
    const a = (text||'').trim();
    const b = (lastBot||'').trim();
    if(b && (a===b || (a.length>20 && b.length>20 && a.startsWith(b.slice(0, Math.min(40,b.length)))))) return;
    lastBot = a;
  }
  const row = document.createElement('div');
  row.className = who === 'You' ? 'row me' : 'row bot';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.innerHTML = (text||'').replace(/</g,'&lt;');
  row.appendChild(b);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function parseInt1(text){
  const m = (text||'').match(/\\d{1,3}/);
  return m ? parseInt(m[0],10) : null;
}

// heuristics for focus anchoring
function looksLikeProblem(text){
  const hasNums = /\\d/.test(text||'');
  const longish = (text||'').length >= 16;
  const mathy = /(total|difference|sum|product|quotient|fraction|percent|rate|area|perimeter|slope|graph|table|equation|x|y)/i.test(text||'');
  return (hasNums && longish) || mathy;
}
function updateFocus(text, imgCount){
  if(/\\bnew question\\b|\\bnext question\\b/i.test(text||'')){ CURRENT = Math.max(1, CURRENT+1); return; }
  if(/\\bnew problem\\b/i.test(text||'')) { FOCUS = ''; return; }
  if(imgCount>0) { FOCUS = '(image problem)'; return; }
  if(looksLikeProblem(text)){ FOCUS = text.slice(0, 300); }
}

async function post(payload){
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json','X-Auth':AUTH},
    body: JSON.stringify({ ...payload, level: LEVEL, total: TOTAL, current: CURRENT, plan_done: PLAN_DONE, focus: FOCUS })
  });
  return r.json();
}

function addThumb(src){
  const d = document.createElement('div'); d.className = 'thumb';
  const img = document.createElement('img'); img.src = src; d.appendChild(img); thumbs.appendChild(d);
}
function fileToDataURL(file){
  return new Promise((res, rej)=>{ const fr = new FileReader(); fr.onload=()=>res(fr.result); fr.onerror=rej; fr.readAsDataURL(file); });
}
fileBtn.onchange = async (e)=>{
  for(const f of e.target.files){ const url = await fileToDataURL(f); queuedImages.push(url); addThumb(url); }
  fileBtn.value = '';
};

unlockBtn.onclick = async ()=>{
  const pw = (pwdBox.value||'').trim(); if(!pw) return;
  addBubble('You','••••••••');
  const data = await post({ message: pw });
  addBubble('MathMate', data.reply ?? data.error ?? '(error)');
  if((data.reply||'').startsWith('🔓')){
    AUTH = pw; unlock.style.display='none'; composer.style.display='flex';
    msgBox.focus();
  }
};

levelSel.onchange = ()=>{ LEVEL = levelSel.value; };
totalSel.onchange = ()=>{ TOTAL = totalSel.value; PLAN_DONE = false; };

sendBtn.onclick = async ()=>{
  const text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;

  // announce 40/50/10 once when we have total + apprentice/rising
  if(!PLAN_DONE && TOTAL && (LEVEL==='Apprentice' || LEVEL==='Rising Hero')){
    addBubble('MathMate', "Here’s our plan 💪  40%: I’ll guide • 50%: you teach me • 10%: I’ll be quiet unless you ask.");
    PLAN_DONE = true;
  } else if(LEVEL==='Master' && !PLAN_DONE){
    addBubble('MathMate', "Okay. You lead—what’s your first move?");
    PLAN_DONE = true;
  }

  updateFocus(text, queuedImages.length);
  addBubble('You', text || '(image(s) only)');
  msgBox.value = ''; sendBtn.disabled = true;
  try{
    const data = await post({ message: text, images: queuedImages });
    addBubble('MathMate', (data.reply ?? data.error ?? '(error)'));
  }finally{
    sendBtn.disabled = false; queuedImages = []; thumbs.innerHTML = ''; msgBox.focus();
  }
};

msgBox.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendBtn.click(); }});
pwdBox.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); unlockBtn.click(); }});
</script>
"""
# ---------- CHAT (vision + meta + anchor) ----------
@app.post("/chat")
def chat():
    try:
        p = request.get_json(silent=True) or {}

        text    = str(p.get("message", "") or "").strip()
        images  = p.get("images") or []
        level   = str(p.get("level", "") or "").strip()
        grade   = str(p.get("grade", "") or "").strip()
        current = str(p.get("current", "") or "").strip()
        focus   = str(p.get("focus", "") or "").strip()

        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! Pick your grade & level, then send your problem or a photo. ✨"), 200
            return jsonify(reply="🔒 Please type the access password to begin."), 200

        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem."}]

        lv = (level or "").lower()
        level_line = ""
        if lv == "apprentice":
            level_line = "LEVEL=Apprentice. You may explain proactively (2–6 short sentences) and must include a guiding question or options."
        elif lv == "rising hero":
            level_line = "LEVEL=Rising Hero. Brief coaching allowed (≤2 short sentences) plus one guiding question or options. Total 1–3 sentences."
        elif lv == "master":
            level_line = "LEVEL=Master. No explanations unless asked. One concise guiding question only."

        grade_line = (
            f"GRADE={grade or 'unknown'} for tone. Use Grade Guide ranges; simplify language for younger grades and increase rigor for older grades."
        )
        focus_line = (
            f"Focus Anchor: {focus or '(infer from latest learner content)'} "
            "Stay on this focus; do not switch topics unless the learner clearly starts a new problem or says 'new question/new problem'."
        )

        def add(msgs, role, content):
            if str(content or "").strip():
                msgs.append({"role": role, "content": content})

        messages = []
        add(messages, "system", MATHMATE_PROMPT)
        add(messages, "system", grade_line)
        add(messages, "system", level_line)
        add(messages, "system", focus_line)
        add(messages, "system", HARD_CONSTRAINT)
        messages.append({"role": "user", "content": user_content})

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            frequency_penalty=0.5,
            presence_penalty=0.2,
            max_tokens=160,
            messages=messages,
        )
        return jsonify(reply=completion.choices[0].message.content)

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        app.logger.exception("Chat crashed: %s", err)
        return jsonify(error=err if DEBUG else "Server error"), 500

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
