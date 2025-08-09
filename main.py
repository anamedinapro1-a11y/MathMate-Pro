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

MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # vision-capable
PASSWORD = os.getenv("MATHMATE_PASSWORD", "unlock-mathmate")
DEBUG    = os.getenv("DEBUG", "0") == "1"

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- TUTOR PROMPT --------------------
MATHMATE_PROMPT = """
MATHMATE — ONE-STEP SOCRATIC TUTOR (Acton + Khan)

GLOBAL RULES (every turn)
• One-Question Rule: ask EXACTLY ONE short question (≤ 2 sentences). No multi-steps.
• Never reveal an operation name or write an equation. Do NOT say add/subtract/multiply/divide, and do NOT write expressions like 19−5.
• Never give the final answer. Never say correct/incorrect. Use neutral acks (“got it”, “noted”) and move on.
• Tone: friendly, concise, 2–3 emojis max, VARY emojis across turns (pool: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔).
• Images: describe only what you SEE (axes, labels, units, fractions/decimals) in a phrase, then ask ONE clarifying question.

LEVELS
• Apprentice (precise + defined): use accurate math words (e.g., sum, difference, product, quotient, factor, multiple, numerator/denominator, variable, expression, equation, inequality, unit rate, slope, intercept, area, perimeter, mean/median/mode, percent). On their FIRST use this session, add a tiny definition in (parentheses), 2–6 words (e.g., “difference (how much bigger)”). After that, use the word normally unless asked.
• Rising Hero: light nudges; still one question.
• Master: bare minimum; one tiny question if possible.

SESSION & PLANNING
• You will receive: level, total_questions, current_question, plan_already_announced.
• If level/total are known, NEVER ask for them again.
• If level is Apprentice or Rising Hero and plan_already_announced is false: announce ONCE “I’ll guide ~40%, you’ll teach back ~50%, last 10% I’ll be here for questions.” (one short sentence + 1–2 emojis), then continue with ONE question. Never repeat it.
• If level is Master: say “Okay.” once when starting, then go minimal.
• Use current_question to pace. If it’s missing/confusing, briefly ask “Which question number are we on now?” and continue next turn.

OUTPUT SHAPE
• One nudge + ONE question ending with “?”. You may include up to 3 short options like “A) …  B) …  C) …”. No equations. No operation names.
"""

HARD_CONSTRAINT = (
    "Hard constraint: reply with ONE short question only (<=2 sentences), "
    "no equations, no operation names, end with a single '?' and nothing after."
)

# -------------------- HEALTH --------------------
@app.get("/health")
def health():
    return "ok", 200

# -------------------- UI (white theme, centered title, bubbles, session controls) --------------------
@app.get("/")
def home():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>MathMate Pro</title>
<style>
  :root{
    --bg:#ffffff; --text:#0f172a; --muted:#64748b; --line:#e2e8f0;
    --me:#e6f0ff; --bot:#f8fafc; --accent:#111827;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial}
  header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:18px 16px;text-align:center}
  header h1{margin:0;font-size:22px;letter-spacing:.2px}
  main{display:flex;justify-content:center}
  .wrap{width:100%;max-width:900px;padding:16px}
  #chat{min-height:58vh;max-height:72vh;overflow:auto;padding:12px 4px}
  .row{display:flex;margin:10px 0}
  .bubble{max-width:72%; padding:12px 14px; border:1px solid var(--line); border-radius:16px; line-height:1.5; white-space:pre-wrap}
  .me{justify-content:flex-end}
  .me .bubble{background:var(--me)}
  .bot{justify-content:flex-start}
  .bot .bubble{background:var(--bot)}
  .sys{color:var(--muted); text-align:center; font-style:italic}
  #panel{position:sticky;bottom:0;background:var(--bg);padding:12px 0;border-top:1px solid var(--line)}
  #unlock{display:flex;gap:8px}
  #password, select, input[type=number]{padding:12px;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--text)}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--line);background:#111827;color:#fff;cursor:pointer;min-width:84px}
  button:disabled{opacity:.6;cursor:not-allowed}
  #composer{display:none;gap:10px;align-items:flex-end;flex-wrap:wrap}
  #left{flex:1;display:flex;flex-direction:column;gap:8px;min-width:300px}
  textarea{flex:1;resize:vertical;min-height:110px;max-height:300px;padding:12px;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--text)}
  #session{display:none;gap:8px;align-items:center;flex-wrap:wrap;border:1px dashed var(--line);border-radius:12px;padding:10px;margin-bottom:8px}
  #drop{border:1px dashed var(--line);border-radius:12px;padding:10px;text-align:center;color:var(--muted)}
  #thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
  .thumb{width:80px;height:80px;border:1px solid var(--line);border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  small.hint{color:var(--muted)}
  .chip{padding:6px 10px;border-radius:12px;border:1px solid var(--line);background:#fff}
</style>

<header><h1>🔒 MathMate Pro</h1></header>
<main><div class="wrap">
  <div id="chat"><div class="sys">Type the password to unlock.</div></div>

  <div id="panel">
    <div id="unlock">
      <input id="password" placeholder="Type password…" />
      <button id="unlockBtn">Unlock</button>
    </div>

    <div id="session">
      <label>Level:
        <select id="levelSel">
          <option value="">choose…</option>
          <option>Apprentice</option>
          <option>Rising Hero</option>
          <option>Master</option>
        </select>
      </label>
      <label>Total questions:
        <input id="totalQ" type="number" min="1" max="100" placeholder="e.g., 7"/>
      </label>
      <label>Current:
        <input id="currentQ" type="number" min="1" max="100" value="1"/>
      </label>
      <button id="prevQ">◀</button>
      <button id="nextQ">▶</button>
      <span class="chip" id="qChip">Q 1 / ?</span>
      <button id="applySession">Apply</button>
      <small class="hint">Set once—MathMate won’t re-ask. Use ◀ ▶ to update question #.</small>
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
</div></main>

<script>
const chat = document.getElementById('chat');
const unlock = document.getElementById('unlock');
const sessionBar = document.getElementById('session');
const composer = document.getElementById('composer');
const msgBox = document.getElementById('msg');
const pwdBox = document.getElementById('password');
const unlockBtn = document.getElementById('unlockBtn');
const sendBtn = document.getElementById('sendBtn');
const fileBtn = document.getElementById('fileBtn');
const drop = document.getElementById('drop');
const thumbs = document.getElementById('thumbs');
const levelSel = document.getElementById('levelSel');
const totalQ = document.getElementById('totalQ');
const currentQ = document.getElementById('currentQ');
const prevQ = document.getElementById('prevQ');
const nextQ = document.getElementById('nextQ');
const applySession = document.getElementById('applySession');
const qChip = document.getElementById('qChip');

let AUTH = '';
let LEVEL = '';
let TOTAL = '';
let CURRENT = '1';
let PLAN_DONE = false;  // announce 40/50/10 only once
let queuedImages = [];

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

function updateChip(){ qChip.textContent = `Q ${CURRENT} / ${TOTAL || "?"}`; }

async function post(payload){
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json','X-Auth':AUTH},
    body: JSON.stringify({
      ...payload,
      level: LEVEL, total: TOTAL, current: CURRENT, plan_done: PLAN_DONE
    })
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
    sessionBar.style.display='flex';
    composer.style.display='flex';
    msgBox.focus();
  }
};

applySession.onclick = ()=>{
  LEVEL = levelSel.value || '';
  TOTAL = (totalQ.value || '').toString();
  CURRENT = (currentQ.value || '1').toString();
  PLAN_DONE = false; // allow the one-time plan announcement after applying
  updateChip();
  addBubble('MathMate', `Session set: Level = ${LEVEL || "?"}, total = ${TOTAL || "?"}, starting at Q ${CURRENT}. ✨`);
  msgBox.focus();
};

prevQ.onclick = ()=>{ const n = Math.max(1, (parseInt(CURRENT||'1')||1) - 1); CURRENT = String(n); currentQ.value = CURRENT; updateChip(); };
nextQ.onclick = ()=>{ const n = (parseInt(CURRENT||'1')||1) + 1; CURRENT = String(n); currentQ.value = CURRENT; updateChip(); };

sendBtn.onclick = async ()=>{
  const text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;
  addBubble('You', text || '(image(s) only)');
  msgBox.value = '';
  sendBtn.disabled = true;
  try{
    const data = await post({ message: text, images: queuedImages });
    addBubble('MathMate', (data.reply ?? data.error ?? '(error)'));
    // after first reply with plan, mark as done so it won't repeat
    if(!PLAN_DONE && LEVEL && TOTAL && (LEVEL === 'Apprentice' || LEVEL === 'Rising Hero')){
      PLAN_DONE = true;
    }
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

# -------------------- CHAT (vision + session meta) --------------------
@app.post("/chat")
def chat():
    try:
        p = request.get_json(silent=True) or {}
        text    = (p.get("message") or "").strip()
        images  = p.get("images") or []
        level   = (p.get("level") or "").strip()
        total   = (p.get("total") or "").strip()
        current = (p.get("current") or "").strip()
        plan_done = bool(p.get("plan_done", False))

        if not text and not images:
            return jsonify(error="Missing 'message' or 'images'"), 400

        # auth
        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! Set your level, total, and current question below to start. ✨")
            return jsonify(reply="🔒 Please type the access password to begin.")

        # assemble user message (vision)
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem."}]

        session_line = (
            f"Session meta: level={level or 'unknown'}; total_questions={total or 'unknown'}; "
            f"current_question={current or 'unknown'}; plan_already_announced={'true' if plan_done else 'false'}. "
            "If level and total are known, do not ask for them again."
        )

        planning_line = ""
        if level and total:
            if level.lower() in ("apprentice", "rising hero", "risinghero"):
                planning_line = (
                    "If plan_already_announced is false, announce the 40/50/10 plan exactly once now; "
                    "otherwise do not mention the plan again."
                )
            else:
                planning_line = "If level is Master, just say 'Okay.' once at start, then be minimal."

        apprentice_define_rule = ""
        if (level or "").lower() == "apprentice":
            apprentice_define_rule = (
                "Apprentice definition rule: when you use a precise math term, include a brief 2–6 word "
                "parenthetical definition on its FIRST appearance this session; do not repeat unless asked."
            )

        messages = [
            {"role": "system", "content": MATHMATE_PROMPT},
            {"role": "system", "content": session_line},
            {"role": "system", "content": planning_line},
            {"role": "system", "content": apprentice_define_rule},
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

# -------------------- LOCAL RUN --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
