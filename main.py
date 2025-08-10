import os, re, time, hashlib
from collections import defaultdict
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

# ---------- LIGHTWEIGHT SESSION MEMORY ----------
# keyed by (level|focus|grade|auth)
MEM = defaultdict(dict)  # {"last_micro": str, "last_sig": str, "ts": float}

def _session_key(level: str, focus: str, grade: str, auth: str) -> str:
    raw = f"{(level or '').lower()}|{(focus or '').strip()}|{(grade or '').lower()}|{auth or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()

def remember_micro(level: str, focus: str, grade: str, auth: str, micro: str):
    key = _session_key(level, focus, grade, auth)
    MEM[key] = {"last_micro": (micro or "").strip(), "last_sig": MEM.get(key, {}).get("last_sig",""), "ts": time.time()}

def last_micro(level: str, focus: str, grade: str, auth: str) -> str:
    return (MEM.get(_session_key(level, focus, grade, auth)) or {}).get("last_micro", "")

def remember_sig(level: str, focus: str, grade: str, auth: str, text: str):
    key = _session_key(level, focus, grade, auth)
    d = MEM.get(key, {})
    d["last_sig"] = hashlib.sha1((text or "").encode()).hexdigest()
    d["ts"] = time.time()
    MEM[key] = d

def last_sig(level: str, focus: str, grade: str, auth: str) -> str:
    return (MEM.get(_session_key(level, focus, grade, auth)) or {}).get("last_sig", "")

# ---------- GRADE BANDS ----------
def grade_band(grade: str) -> str:
    """
    Returns: K-2, 3-5, 6-8, 9-12
    Accepts 'K', '2', '2nd', 'grade 7', '11', etc.
    """
    g = (grade or "").strip().lower()
    if g in {"k", "kindergarten", "kg"}:
        n = 0
    else:
        m = re.search(r"\d+", g)
        n = int(m.group()) if m else 8
        n = min(max(n, 0), 12)
    if 0 <= n <= 2:  return "K-2"
    if 3 <= n <= 5:  return "3-5"
    if 6 <= n <= 8:  return "6-8"
    return "9-12"

# ---------- SYSTEM PROMPTS ----------
# ---------- MATHMATE PROMPT ----------
MATHMATE_PROMPT = """
🎯 MATHMATE – ACTON + KHAN ACADEMY AI GUIDE (Socratic)

ROLE
You are a Socratic math guide (not a teacher), Acton Academy style, for learners age 13+. You NEVER give answers. You help students discover them through questions, doubt, and teach-back.

HOW TO RESPOND
• ✅ Only ask QUESTIONS or offer OPTIONS. Replies are short (1–3 sentences).
• ✅ You MAY name operations **inside a question or options list only** (e.g., “A) Add  B) Subtract  C) Multiply  D) Divide”). Never state an operation as an instruction.
• ❌ Never say or imply “correct/incorrect” or “you’re right”.
• ❌ Never explain unless asked directly.
• ✅ Nudge with: “Try it out!”, “Want to test that with the graph?”, “What makes you confident?”

WHEN THE LEARNER PICKS AN OPTION
• If the learner chooses one (e.g., “B) Subtract” or types “subtract”), do NOT re-ask the options.
• Acknowledge by reframing as a question and move one step forward, e.g.:
  “Cool—subtracting compares amounts; which two numbers will you compare first, and in what order?” (still a question; no confirmation).

KHAN SCREENSHOT RULES
• Check format: fraction vs decimal; which is x vs y; any graph present?
• If graph: ask for a clear point; ask which axis is which; ask what happens with y/x (unit rate).
• If thinking is right but format is off: “Does Khan want decimal or fraction?”

CHALLENGE LEVELS
• 🐣 Apprentice — slow, define needed terms, clear step-by-step questions, patient, never give full answers.
• 🦸 Rising Hero — light support only (≤4 short sentences), ask one helpful question.
• 🧠 Master — say as little as possible; “What’s your first step?”

QUIZ STRATEGY (40/50/10)
• If total_questions is known and level ∈ {Apprentice, Rising Hero} and plan not yet announced:
  Say ONCE: “Here’s our plan 💪  40%: I’ll guide • 50%: you teach me • 10%: I’ll be quiet unless you ask.”
• Ask the learner to tell you when they start a new question so pacing matches the plan.

ANCHORING (critical)
• You will receive a Focus Anchor describing the current problem.
• STAY on that focus; do not switch topics unless the learner clearly starts a new problem or says “new question/new problem”.
• If the learner says “I don’t know”, keep the focus and ask a smaller clarifying question or offer 2–3 options.

LOOP GUARD (no repeats)
• Offer the A/B/C/D operation menu at most ONCE per question unless the learner asks to go back.
• Do not reuse the same sentence stem twice in a row; vary wording each turn.
• Do not repeat “First, say what you’re trying to find …” or re-state general instructions after they’ve been shown once.
• If your last message contained options, the next message MUST be a single follow-up question that advances the step.

STYLE
• Friendly, respectful, curious; never condescending.
• Vary emojis (max 2) from: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔.
• Keep ≤1 question mark per reply and ≤2 sentences total.
"""

HARD_CONSTRAINT = (
    "Hard constraint: respond ONLY with questions or short option sets; "
    "no answers, no correctness; you MAY name operations only inside questions/options; "
    "<= 2 sentences total and a single '?'; do not repeat prior option menus; stay anchored to the provided focus."
)

# ---------- HUMAN-LIKE PHRASE BANKS ----------
SCAFFOLD_K2_FIRST = [
    "First, say what you’re trying to find. 😊",
    "First, point to what the question wants.",
    "First, tell me the goal in your own words.",
]
SCAFFOLD_K2_NEXT = [
    "Then, match each number to a person or thing.",
    "Then, line up the two amounts you’re comparing.",
    "Then, think: who has more and who has less?",
]
SCAFFOLD_UP_FIRST = [
    "First, name what’s being asked.",
    "Start by stating the goal.",
    "Begin by saying what you’re trying to find.",
]
SCAFFOLD_UP_NEXT = [
    "Then, match each number to a role (x vs y or who/what).",
    "Then, line up the two amounts you’re comparing.",
    "Then, decide how the two quantities relate.",
]
REVIEW_PROMPTS = [
    "Mmm, let’s review the steps.",
    "Let’s check again—",
    "Let’s step back for a sec.",
]
REFLECTIVE_QS = [
    "What makes you confident that fits here?",
    "What tells you that matches the question?",
    "How does that connect to what’s being asked?",
]

def _pick(seq, seed: str) -> str:
    if not seq: return ""
    h = int(hashlib.sha1((seed or '').encode()).hexdigest(), 16)
    return seq[h % len(seq)]

# ---------- OUTPUT GUARDRAILS ----------
_CONFIRM_WORDS = re.compile(r"\b(correct|right|exactly|nailed\s*it|that\s*works|you'?re\s*right|spot\s*on)\b", re.I)
_EQN_BITS = re.compile(r"([=+\-*/^]|(?<!\w)%|\b\d+\s*/\s*\d+\b)")
_COORDS   = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")
_ANSWER_PH = re.compile(r"\b(the answer is|therefore|thus|equals|so you get)\b", re.I)
_TAG_OK    = "[[LIKELY_OK]]"
_TAG_OFF   = "[[LIKELY_OFF]]"

# ---- Options builder / operation detection ----
def _needs_options(text: str) -> bool:
    return not re.search(r"(^|\n)\s*(A\)|B\)|C\)|•|-)\s+", text)

def _op_options_for_focus(focus: str, band: str) -> str:
    f = (focus or "").lower()
    if any(w in f for w in ["fewer", "more", "less", "difference", "compare", "how many left", "take away", "left", "remain"]):
        return "A) Add (put together)  B) Subtract (take away)"
    if any(w in f for w in ["groups of", "each has", "same size groups", "area", "rectangle", "times"]):
        return "A) Add repeatedly  B) Multiply"
    if any(w in f for w in ["per", "each", "rate", "split", "share", "equal groups", "average"]):
        return "A) Multiply  B) Divide"
    return "A) Add  B) Subtract  C) Multiply  D) Divide"

_OP_WORDS = re.compile(r"\b(add|plus|sum|subtract|minus|difference|take\s*away|fewer|less|more than|multiply|times|product|divide|divided\s+by|over|quotient|rate|per|each|share)\b", re.I)

def _operation_question_for_focus(focus: str, band: str) -> str:
    f = (focus or "").lower()
    if any(w in f for w in ["fewer", "less", "difference", "compare", "take away", "left", "remain", "more than"]):
        return "Do we add or subtract?"
    if any(w in f for w in ["per", "each", "rate", "share", "split", "quotient", "over"]):
        return "Should we multiply or divide?"
    if any(w in f for w in ["groups of", "times", "same size groups", "area", "rectangle"]):
        return "Is this multiplying or adding repeatedly?"
    return "Which operation fits best?"

def _force_operation_choice(text: str, focus: str, level: str, band: str) -> str:
    """
    If the message mentions operations/synonyms, ensure we end with a clean
    operation-choice question + options (no vague 'what would you try next?').
    """
    t = (text or "").strip()
    contains_ops = bool(_OP_WORDS.search(t))
    if not contains_ops:
        return t

    # Remove narrations like "take away 5 from 12"
    t = re.sub(r"\btake\s+away\s+[^\.!?]+?\s+from\s+[^\.!?]+", "compare the two amounts", t, flags=re.I)

    # Replace any trailing question with an operation-choice question
    op_q = _operation_question_for_focus(focus, band)
    if "?" in t:
        # keep everything before last sentence, replace last with op_q
        last_q = t.rfind("?")
        start = t.rfind(".", 0, last_q)
        start = 0 if start == -1 else start + 1
        prefix = t[:start].rstrip(".!… ")
        t = (prefix + (". " if prefix else "") + op_q).strip()
    else:
        t = (t.rstrip(".!…") + ". " + op_q).strip()

    if _needs_options(t):
        t += "\n" + _op_options_for_focus(focus, band)
    return t

def _split_micro_and_question(text: str):
    q_idx = text.rfind("?")
    if q_idx == -1:
        return text.strip(), ""
    micro = text[:q_idx].strip()
    start = text.rfind(".", 0, q_idx)
    start = 0 if start == -1 else start + 1
    question = text[start:q_idx+1].strip()
    return micro, question

def _limit_form(text: str, band: str, level: str, focus: str) -> str:
    """
    Trim sentences and, if we somehow still don't end with a specific question,
    convert to a clear choice question (never 'what would you try next?').
    """
    max_sent = 3 if band == "K-2" else 4
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()][:max_sent]
    t = " ".join(parts) if parts else ""
    # Ensure exactly one '?'
    qs = [m.start() for m in re.finditer(r"\?", t)]
    if len(qs) == 0:
        # No question → make an operation-choice or generic choice question with options
        op_q = _operation_question_for_focus(focus, band)
        t = (t.rstrip(".!…") + (". " if t else "") + op_q).strip()
    elif len(qs) > 1:
        last = qs[-1]
        buff = []
        for i, ch in enumerate(t):
            buff.append("." if ch == "?" and i != last else ch)
        t = "".join(buff)
    # Apprentice always sees choices
    if (level or "").lower() == "apprentice" and _needs_options(t):
        t += "\n" + _op_options_for_focus(focus, band)
    return t

def _simplify_for_k2(text: str) -> str:
    swaps = {
        r"\bdetermine\b": "find",
        r"\bidentify\b": "find",
        r"\bcompare\b": "look at",
        r"\bquantity\b": "amount",
        r"\bconsider\b": "think about",
        r"\bstatement\b": "sentence",
    }
    for pat, rpl in swaps.items():
        text = re.sub(pat, rpl, text, flags=re.I)
    text = re.sub(r",\s+and", " and", text)
    return text

def _pick_scaffold(level: str, band: str, seed: str, user_answer_like: bool) -> str:
    # Do NOT scaffold right after a numeric answer; it feels off-topic.
    if (level or "").lower() != "apprentice" or user_answer_like:
        return ""
    if band == "K-2":
        first = _pick(SCAFFOLD_K2_FIRST, seed)
        second = _pick(SCAFFOLD_K2_NEXT, "next-"+seed)
    else:
        first = _pick(SCAFFOLD_UP_FIRST, seed)
        second = _pick(SCAFFOLD_UP_NEXT, "next-"+seed)
    return f"{first} {second}"

def enforce_mathmate_style(text: str, level: str, focus: str, grade: str, auth: str, user_answer_like: bool) -> str:
    band = grade_band(grade)

    # hidden tag
    tag = "OFF"
    if text and _TAG_OK in text:
        tag, text = "OK", text.replace(_TAG_OK, "")
    elif text and _TAG_OFF in text:
        tag, text = "OFF", text.replace(_TAG_OFF, "")
    t = (text or "").strip()

    # clean confirmations / answer-y claims; hide equations/operators
    t = _CONFIRM_WORDS.sub(" ", t)
    t = _ANSWER_PH.sub("What makes you confident", t)
    t = _EQN_BITS.sub("…", t)
    t = _COORDS.sub("that point", t)

    # de-dupe micro-lesson
    micro, question = _split_micro_and_question(t)
    prev_micro = last_micro(level, focus, grade, auth)
    if micro and prev_micro and micro.strip().lower() == prev_micro.strip().lower():
        t = question or ""
    else:
        if micro: remember_micro(level, focus, grade, auth, micro)

    # Apprentice scaffolding (grade-aware; skip if user just gave a number)
    scaff = _pick_scaffold(level, band, focus, user_answer_like)
    if scaff and scaff not in t:
        t = (scaff + " " + t).strip()

    # Grade simplifier
    if band == "K-2":
        t = _simplify_for_k2(t)

    # ALWAYS convert operation-y phrasing into a clean question + options
    t = _force_operation_choice(t, focus, level, band)

    # shape & single question; never “what would you try next?”
    t = _limit_form(t, band, level, focus)

    # Answer encouragement policy (ONLY if learner proposed a lone value)
    if user_answer_like:
        if tag == "OK":
            if not t.lstrip().startswith("✅ Try it"):
                t = "✅ Try it. " + t
        else:
            if not (t.lstrip().startswith("Mmm, let’s review the steps.") or t.lstrip().startswith("Let's check again—") or t.lstrip().startswith("Let’s step back for a sec.") ):
                t = _pick(REVIEW_PROMPTS, focus) + " " + t

    # reduce exact repeats
    sig = hashlib.sha1(t.encode()).hexdigest()
    if sig == last_sig(level, focus, grade, auth):
        base = re.sub(r"\?.*$", "", t).rstrip(".!…")
        alt_q = _pick(REFLECTIVE_QS, 'reflect-'+focus)
        t = f"{base}. {alt_q}?"
    remember_sig(level, focus, grade, auth, t)
    return t.strip()

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
  .wrap{width:100%;max-width:980px;padding:16px}
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
  input,button,select{font:inherit}
  #password, textarea, select{padding:12px;border-radius:12px;border:1px solid var(--line);background:#fff;color:#0f172a}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--line);background:#111827;color:#fff;cursor:pointer;min-width:84px}
  button:disabled{opacity:.6;cursor:not-allowed}
  #composer{display:none;gap:10px;align-items:flex-end;flex-wrap:wrap}
  #left{flex:1;display:flex;flex-direction:column;gap:8px;min-width:320px}
  #topline{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  textarea{flex:1;resize:vertical;min-height:110px;max-height:300px}
  #drop{border:1px dashed var(--line);border-radius:12px;padding:10px;text-align:center;color:var(--muted)}
  #thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
  .thumb{width:80px;height:80px;border:1px solid var(--line);border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  small.hint{color:var(--muted)}
  label{font-size:14px;color:#334155}
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
        <div id="topline">
          <label>Grade:
            <select id="grade">
              <option value="">Select grade…</option>
              <option value="K">K</option>
              <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
              <option>6</option><option>7</option><option>8</option><option>9</option><option>10</option><option>11</option><option>12</option>
            </select>
          </label>
          <label>Level:
            <select id="levelSel">
              <option value="">Pick level…</option>
              <option>Apprentice</option>
              <option>Rising Hero</option>
              <option>Master</option>
            </select>
          </label>
        </div>
        <textarea id="msg" placeholder="Send your problem or a photo. (Shift+Enter = newline)"></textarea>
        <div id="drop">
          <label for="fileBtn">➕ Add images (PNG/JPG) — drag & drop or click</label>
          <input id="fileBtn" type="file" accept="image/*" multiple />
          <div id="thumbs"></div>
          <small class="hint">Tip: choose grade + level first. Say “new problem” to switch topics.</small>
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
const gradeSel = document.getElementById('grade');
const levelSel = document.getElementById('levelSel');

let AUTH = '';
let LEVEL = '';       // Apprentice | Rising Hero | Master
let GRADE = '';       // K | 1..12
let FOCUS = '';       // sticky anchor
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

function looksLikeProblem(text){
  const hasNums = /\\d/.test(text||'');
  const longish = (text||'').length >= 16;
  const mathy = /(total|difference|sum|product|quotient|fraction|percent|area|perimeter|slope|graph|points|solve|x|y|how many|fewer|more|less|groups|each|rate|per|left|remain)/i.test(text||'');
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
    body: JSON.stringify({ ...payload, level: LEVEL, focus: FOCUS, grade: GRADE })
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
    addBubble('MathMate', "Pick your grade and level, then send your problem or a photo. ✨");
    msgBox.focus();
  }
};

levelSel.onchange = ()=>{ LEVEL = levelSel.value; };
gradeSel.onchange = ()=>{ GRADE = gradeSel.value; };

sendBtn.onclick = async ()=>{
  let text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;
  if(!AUTH) return;

  if(!LEVEL || !GRADE){
    addBubble('MathMate', "Choose a **grade** and a **level** first (top of the box), then send your problem. 🙂");
    return;
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

# ---------- CHAT (vision + level + focus + grade) ----------
@app.post("/chat")
def chat():
    try:
        p = request.get_json(silent=True) or {}
        text   = (p.get("message") or "").strip()
        images = p.get("images") or []
        level  = (p.get("level") or "").strip()
        focus  = (p.get("focus") or "").strip()
        grade  = (p.get("grade") or "").strip()

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

        # Hints
        band = grade_band(grade)
        grade_line = (
            f"Grade band: {band}. Use language and sentence length that fit this band; "
            "for K-2 keep sentences very short and concrete; use friendly tone and simple words."
        )
        session_line = (
            f"Session meta: level={level or 'unknown'}, grade={grade or 'unknown'}. "
            "If level is present, do not ask for it again; start guiding immediately."
        )
        focus_line = (
            f"Focus Anchor: {focus or '(no explicit anchor; infer from last user message/image)'} "
            "Stay on this focus and do not switch topics unless the learner clearly starts a new problem or says 'new problem'. "
            "If the learner says 'I don’t know', provide a tiny micro-lesson relevant to THIS focus and ask a smaller clarifying question."
        )

        style_line = ""
        lv = (level or "").lower()
        if lv == "apprentice":
            style_line = "Apprentice: include 1–2 scaffold lines (natural verbs; no arithmetic) before your single question, and add options if helpful."
            if band == "K-2":
                style_line += " Use kid-friendly words and keep sentences short."
        elif lv == "rising hero":
            style_line = "Rising Hero: one short nudge if needed, then your single question."
        elif lv == "master":
            style_line = "Master: minimal; ask one question only."

        # Did user just propose a single value? (e.g., "8")
        user_answer_like = bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", text))

        messages = [
            {"role": "system", "content": MATHMATE_PROMPT},
            {"role": "system", "content": GUIDE_RULES},
            {"role": "system", "content": HARD_CONSTRAINT},
            {"role": "system", "content": grade_line},
            {"role": "system", "content": focus_line},
            {"role": "system", "content": session_line},
            {"role": "system", "content": style_line},
            {"role": "user", "content": user_content},
        ]

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=messages,
        )
        raw = completion.choices[0].message.content
        auth = request.headers.get("X-Auth","")
        reply = enforce_mathmate_style(raw, level, focus, grade, auth, user_answer_like)
        return jsonify(reply=reply)

    except Exception as e:
        if DEBUG:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
