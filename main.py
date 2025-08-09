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

# ---------- OUTPUT GUARDRAILS (questions/options only + no confirmations) ----------
_CONFIRM_WORDS = re.compile(r"\b(correct|right|exactly|nailed\s*it|that\s*works|you'?re\s*right)\b", re.I)
# hide explicit operation hints
_OP_NAMES = re.compile(r"\b(add|subtract|multiply|divide|plug|replace|simplify|distribute|factor|solve|isolate|cross[-\s]?multiply)\b", re.I)
# raw operators/equations/inline fractions
_EQN_BITS = re.compile(r"([=+\-*/^]|(?<!\w)%|\b\d+\s*/\s*\d+\b)")
# coordinates like (4, 2.5)
_COORDS = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")
# answer-revealy phrases
_EXPLAINERS = re.compile(r"\b(the answer is|so you get|therefore|thus|equals)\b", re.I)

def _limit_sentences_and_questions(text: str) -> str:
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()][:3]  # ≤3 sentences
    text = " ".join(parts) if parts else ""

    # exactly one '?'
    q_positions = [m.start() for m in re.finditer(r"\?", text)]
    if len(q_positions) == 0:
        text = (text.rstrip(".!…") + " — what do you want to try next?").strip() if text else "What do you want to try first?"
    elif len(q_positions) > 1:
        last = q_positions[-1]
        buff = []
        for i, ch in enumerate(text):
            buff.append("." if ch == "?" and i != last else ch)
        text = "".join(buff)
    return text

def enforce_mathmate_style(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "What do you want to try first?"

    # strip confirmations
    t = _CONFIRM_WORDS.sub(" ", t)
    # avoid giving away operations
    t = _OP_NAMES.sub("that step", t)
    # hide raw equations/operators
    t = _EQN_BITS.sub("…", t)
    # avoid explicit coordinates
    t = _COORDS.sub("that point", t)
    # avoid declarative “here’s the answer”
    if _EXPLAINERS.search(t) and "?" not in t:
        t = re.sub(_EXPLAINERS, "What makes you confident that", t)

    t = _limit_sentences_and_questions(t)

    # ensure question/options ending
    if "?" not in t and not re.search(r"(^|\n)\s*([\-•]|A\)|1\))", t):
        t = t.rstrip(".!…") + " — what would you try next?"
    return t.strip()

# ---------- TUTOR PROMPT (anchored; your original) ----------
MATHMATE_PROMPT = """
MATHMATE — SOCRATIC TUTOR with MICRO-LESSONS (Acton + Khan)

ANCHORING RULES (very important)
• You will receive a Focus Anchor describing the current problem (numbers/scene/user text).
• STAY on this focus. Do not switch topics or introduce new concepts/examples unless the learner clearly starts a new problem or says “new problem”.
• If the learner says “I don’t know”, give a micro-lesson relevant to the current focus and ask a smaller clarifying question—do not change topics.

GLOBAL STYLE
• Teach-while-asking: MICRO-LESSON first (transferable idea/definition/pattern/pitfall), then ONE question.
• Micro-lesson is brief and reusable; do NOT solve the problem or name the operation.
• One-Question Rule: ask EXACTLY ONE question (1 sentence). No lists, no multi-steps, only one “?” total.
• Never reveal an operation or write an equation. Do NOT say add/subtract/multiply/divide. Do NOT write expressions like 19−5.
• Never give the final answer. Never say correct/incorrect. Use neutral acks (“got it”, “noted”).
• Friendly + concise + 2–3 varied emojis (pool: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔).
• Images: briefly state what you SEE (axes, labels, units, fractions/decimals) in a phrase, then micro-lesson + ONE question.

LEVELS
• Apprentice (precise + defined): use accurate math terms (sum, difference, product, quotient, factor, multiple, numerator/denominator, variable, expression, equation, inequality, rate, slope, intercept, area, perimeter, mean/median/mode, percent). On FIRST use this session, add a 2–6 word parenthesis definition, e.g., “quotient (result of division)”.
• Rising Hero: micro-lesson only if needed (≤1 sentence). Light nudge.
• Master: minimal. No micro-lesson unless asked.

SESSION
• You will receive: level and focus_anchor. If level is present, never ask for it again. If focus_anchor is present, do not change topics away from it.

OUTPUT SHAPE
• MICRO-LESSON (0–2 short statements, no “?”) → ONE question ending with “?”.
• Up to 3 short options allowed (e.g., “A) …  B) …  C) …”).
• Absolutely no equations and no operation names.
"""

# ---------- EXTRA GUIDE RULES (merged; 40/50/10 REMOVED) ----------
GUIDE_RULES = """
You are MathMate — a Socratic math GUIDE (not a teacher) for learners 13+, inspired by Acton Academy and using Khan Academy-style problems.

RESPONDING
- Aside from brief micro-lesson lines, respond ONLY with QUESTIONS or concise OPTION lists.
- Never say or imply “correct,” “right,” or confirm correctness.
- Do not explain further unless the learner directly asks.

WHEN THE LEARNER PROPOSES AN ANSWER
- Do NOT confirm it. Nudge thinking with:
  • “Try it out.”
  • “What made you choose that?”

IF THE LEARNER IS STUCK OR OFF-TRACK
- Ask process questions:
  • “What step did you try first?”
  • “Can you walk me through your thinking?”
  • Graphs: “Does that point match the graph?”
  • Equations: “What’s your first move?”
  • Tables: “Are the numbers consistent?”

SCREENSHOTS / FORMAT CHECKS
- Clarify early:
  • “Is the answer needed as a fraction or decimal?”
  • “Which is x and which is y?”
  • “Is there a graph? Can you find a clean point?”
  • “What happens when you divide y by x?”
- If reasoning is right but format mismatched:
  • “Great thinking — does Khan want that as a decimal or a fraction?”

MATH ACCURACY
- Use a calculator/tool for arithmetic when needed; do not guess.
- Double-check x vs y.
- Match the exact output format Khan requests.
"""

HARD_CONSTRAINT = (
    "Hard constraint: output a micro-lesson first (0–2 short statements, no '?'), "
    "then EXACTLY ONE question (1 sentence) — total ≤ 3 sentences and only one '?'. "
    "No equations. No operation names. Stay anchored to the provided focus."
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
        <textarea id="msg" placeholder="Tell me your level (Apprentice / Rising Hero / Master), then send your problem or a photo. (Shift+Enter = newline)"></textarea>
        <div id="drop">
          <label for="fileBtn">➕ Add images (PNG/JPG) — drag & drop or click</label>
          <input id="fileBtn" type="file" accept="image/*" multiple />
          <div id="thumbs"></div>
          <small class="hint">Images are analyzed with your prompt (vision). Say “new problem” to switch topics.</small>
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
let LEVEL = '';       // Apprentice | Rising Hero | Master
let FOCUS = '';       // sticky anchor text for the current problem
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

function pickLevelFrom(text){
  const t = (text||'').toLowerCase();
  if(t.includes('apprentice')) return 'Apprentice';
  if(t.includes('rising hero')) return 'Rising Hero';
  if(t.includes('master')) return 'Master';
  return '';
}

function looksLikeProblem(text){
  const hasNums = /\\d/.test(text||'');
  const longish = (text||'').length >= 16;
  const mathy = /(total|difference|sum|product|quotient|fraction|percent|area|perimeter|slope|graph|points|solve|x|y)/i.test(text||'');
  return (hasNums && longish) || mathy;
}

function resetFocusIfNewProblem(text, imgCount){
  if(/\\bnew problem\\b/i.test(text||'')) { FOCUS = ''; return; }
  if(imgCount > 0) { FOCUS = '(image problem)'; return; }
  if(looksLikeProblem(text)) { FOCUS = text.slice(0, 300); }
}

async function post(payload){
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json','X-Auth':AUTH},
    body: JSON.stringify({ ...payload, level: LEVEL, focus: FOCUS })
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
    addBubble('MathMate', "Which level should we use—🐣 Apprentice, 🦸 Rising Hero, or 🧠 Master?");
    msgBox.focus();
  }
};

sendBtn.onclick = async ()=>{
  let text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;

  if(!LEVEL){
    addBubble('You', text || '(image(s) only)');
    const lv = pickLevelFrom(text);
    if(lv){
      LEVEL = lv;
      addBubble('MathMate', `Great — we’ll use **${LEVEL}** mode. Send your problem or a photo. ✨`);
    }else{
      addBubble('MathMate', "Please choose: Apprentice, Rising Hero, or Master. 🙂");
    }
    msgBox.value = ''; return;
  }

  resetFocusIfNewProblem(text, queuedImages.length);

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

# ---------- CHAT (vision + level + focus) ----------
@app.post("/chat")
def chat():
    try:
        p = request.get_json(silent=True) or {}
        text   = (p.get("message") or "").strip()
        images = p.get("images") or []
        level  = (p.get("level") or "").strip()
        focus  = (p.get("focus") or "").strip()  # sticky anchor

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
        focus_line = (
            f"Focus Anchor: {focus or '(no explicit anchor; infer from last user message/image)'} "
            "Stay on this focus and do not switch topics unless the learner clearly starts a new problem or says 'new problem'. "
            "If the learner says 'I don’t know', provide a micro-lesson relevant to THIS focus and ask a smaller clarifying question."
        )

        apprentice_define_rule = ""
        if (level or "").lower() == "apprentice":
            apprentice_define_rule = (
                "Apprentice rule: when you use a precise math term, include a brief 2–6 word "
                "parenthetical definition on its FIRST appearance this session; do not repeat unless asked."
            )
        intensity_line = ""
        if (level or "").lower() == "rising hero":
            intensity_line = "Rising Hero style: add a tiny micro-lesson only if needed; one light nudge."
        elif (level or "").lower() == "master":
            intensity_line = "Master style: minimal; no micro-lesson unless asked; one tiny question."

        messages = [
            {"role": "system", "content": MATHMATE_PROMPT},
            {"role": "system", "content": GUIDE_RULES},   # merged GUIDE (no 40/50/10)
            {"role": "system", "content": focus_line},
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
        raw = completion.choices[0].message.content
        return jsonify(reply=enforce_mathmate_style(raw))

    except Exception as e:
        if DEBUG:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
