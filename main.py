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
# keyed by (level|focus|auth)
MEM = defaultdict(dict)  # {"last_micro": str, "last_sig": str, "ts": float}

def _session_key(level: str, focus: str, auth: str) -> str:
    raw = f"{(level or '').lower()}|{(focus or '').strip()}|{auth or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()

def remember_micro(level: str, focus: str, auth: str, micro: str):
    key = _session_key(level, focus, auth)
    d = MEM.get(key, {})
    d["last_micro"] = (micro or "").strip()
    d["ts"] = time.time()
    MEM[key] = d

def last_micro(level: str, focus: str, auth: str) -> str:
    return (MEM.get(_session_key(level, focus, auth)) or {}).get("last_micro", "")

def remember_sig(level: str, focus: str, auth: str, text: str):
    key = _session_key(level, focus, auth)
    d = MEM.get(key, {})
    d["last_sig"] = hashlib.sha1((text or "").encode()).hexdigest()
    d["ts"] = time.time()
    MEM[key] = d

def last_sig(level: str, focus: str, auth: str) -> str:
    return (MEM.get(_session_key(level, focus, auth)) or {}).get("last_sig", "")

# ---------- SYSTEM PROMPTS ----------
MATHMATE_PROMPT = """
MATHMATE — SOCRATIC GUIDE (Acton + Khan). You guide ages 13+. You never give final answers.

ANCHORING
• You receive a Focus Anchor for the current problem. Stay on it unless the learner clearly starts a new one (“new problem”).
• If the learner says “I don’t know,” give a tiny micro-lesson for THIS focus, then a smaller question.

STYLE
• Natural, warm, and clear. Question-led. Respectful and curious — never condescending.
• Do not say “correct/incorrect” or otherwise confirm correctness.
• Screenshots: briefly check format (fraction vs decimal, x vs y), note any graph and a clean point.

LEVELS
• Apprentice — give 1–2 short scaffolding lines using natural verbs (no arithmetic), define the first math term briefly, then ask ONE question.
• Rising Hero — one short nudge if needed, then ONE question.
• Master — minimal; ask ONE question only.

OUTPUT SHAPE
• Micro-lesson (0–2 brief statements) and/or scaffolding (≤2 lines) → EXACTLY ONE question (one “?” total), ≤4 sentences overall.
• Avoid explicit operation names like add/subtract/multiply/divide and avoid equations or operator symbols in your wording.
"""

GUIDE_RULES = """
RESPONDING
• Aside from the tiny micro-lesson/scaffold lines, respond ONLY with QUESTIONS or concise OPTION lists.
• Do not give final answers or explicit arithmetic instructions.

WHEN THE LEARNER PROPOSES AN ANSWER
• Make a best-effort internal judgment: LIKELY_OK vs LIKELY_OFF (do not reveal the judgment).
• If LIKELY_OK: start with “✅ Try it.” then a reflective question.
• If LIKELY_OFF/UNCLEAR: do NOT encourage entering it; start with “Mmm, let’s review the steps.” or “Let’s check again—” then a guiding question.

FORMAT/SCREENSHOTS
• Ask early: “fraction or decimal?”, “which is x and which is y?”, “is there a graph — can you find a clear point?”, “what happens when you divide y by x?”

HIDDEN TAG (required)
• Append exactly [[LIKELY_OK]] or [[LIKELY_OFF]] at the very end of EVERY reply, based on your private judgment about any just-proposed value; if no value was proposed, use [[LIKELY_OFF]]. Do not explain this tag.
"""

HARD_CONSTRAINT = (
    "Hard constraint: micro-lesson first (0–2 short statements), optionally 1–2 scaffold lines (natural verbs only), "
    "then EXACTLY ONE question (one '?'). ≤4 sentences total. "
    "Avoid operation names and equations. Stay anchored to the provided focus."
)

# ---------- HUMAN-LIKE PHRASE BANKS ----------
SCAFFOLD_STARTS = [
    "First, name what’s being asked.",
    "Start by naming the goal.",
    "Begin by saying what you’re trying to find.",
]
SCAFFOLD_NEXT = [
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
    "What makes you confident that approach fits here?",
    "What tells you that matches the question?",
    "How does that connect to what’s being asked?",
]
GUIDING_QS = [
    "When you compare them, what are you finding: how many more, or how many fewer?",
    "Which quantity should you start from to match the wording?",
    "What labels would you put on each number to make the comparison clear?",
]

def _pick(seq, seed: str) -> str:
    # deterministic variety based on the focus/session
    if not seq:
        return ""
    h = int(hashlib.sha1((seed or '').encode()).hexdigest(), 16)
    return seq[h % len(seq)]

# ---------- OUTPUT GUARDRAILS ----------
_CONFIRM_WORDS = re.compile(r"\b(correct|right|exactly|nailed\s*it|that\s*works|you'?re\s*right|spot\s*on)\b", re.I)

# operation tokens (verbs & keywords); we won't blindly replace; we rewrite clauses gracefully
_OP_TOKENS = re.compile(
    r"\b(add|plus|sum|subtract|minus|difference|multiply|times|product|divide|divided\s+by|over|quotient)\b",
    re.I,
)

# mask explicit arithmetic/operators and coordinate pairs (we’ll rewrite wording first)
_EQN_BITS = re.compile(r"([=+\-*/^]|(?<!\w)%|\b\d+\s*/\s*\d+\b)")
_COORDS   = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")
_ANSWER_PH = re.compile(r"\b(the answer is|therefore|thus|equals|so you get)\b", re.I)

_TAG_OK    = "[[LIKELY_OK]]"
_TAG_OFF   = "[[LIKELY_OFF]]"

def _extract_hidden_tag(text: str):
    if _TAG_OK in text:
        return "OK", text.replace(_TAG_OK, "")
    if _TAG_OFF in text:
        return "OFF", text.replace(_TAG_OFF, "")
    return "OFF", text  # default safe

def _split_micro_and_question(text: str):
    q_idx = text.rfind("?")
    if q_idx == -1:
        return text.strip(), ""
    micro = text[:q_idx].strip()
    start = text.rfind(".", 0, q_idx)
    start = 0 if start == -1 else start + 1
    question = text[start:q_idx+1].strip()
    return micro, question

def _limit_form(text: str) -> str:
    # ≤4 sentences; exactly one '?'
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()][:4]
    text = " ".join(parts) if parts else ""
    qs = [m.start() for m in re.finditer(r"\?", text)]
    if len(qs) == 0:
        text = (text.rstrip(".!…") + " — what would you try next?").strip() if text else "What would you try first?"
    elif len(qs) > 1:
        last = qs[-1]
        buff = []
        for i, ch in enumerate(text):
            buff.append("." if ch == "?" and i != last else ch)
        text = "".join(buff)
    return text

# --- Grammar-aware operation rewrites (natural English) ---
def _normalize_ops_phrasing(text: str) -> str:
    # 1) “subtract X from Y” → “find the difference between Y and X”
    text = re.sub(
        r"\b(subtract)\s+([^\.?\n]+?)\s+(from)\s+([^\.?\n]+?)([\.\?!])",
        lambda m: f"find the difference between {m.group(4)} and {m.group(2)}{m.group(5)}",
        text,
        flags=re.I,
    )
    # 2) “Y minus X” in a clause (not necessarily the question)
    text = re.sub(
        r"\b([A-Za-z0-9 _]+?)\s+minus\s+([A-Za-z0-9 _]+?)\b",
        r"the difference between \1 and \2",
        text,
        flags=re.I,
    )
    # 3) Simplify other verbs into gentle, non-robotic phrases
    replacements = {
        r"\badd\b": "combine",
        r"\bplus\b": "combine",
        r"\bmultiply\b": "scale",
        r"\btimes\b": "scale",
        r"\bdivide\b": "compare as a rate",
        r"\bdivided\s+by\b": "compare as a rate",
        r"\bover\b": "compare as a rate",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text, flags=re.I)

    # 4) If the final question still mentions an explicit operation with two numerals, make it generic.
    q_idx = text.rfind("?")
    if q_idx != -1:
        before = text[:q_idx]
        question = text[q_idx-220 if q_idx-220>0 else 0:q_idx+1]
        if re.search(r"\b(\d+)\s*(minus|plus|times|divided\s+by|over)\s*(\d+)\b", question, re.I):
            # Use a human, non-leading question
            question = "How many more (or fewer) is that?"
            text = (before.strip() + " " + question).strip()
    # Clean spacing
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def _vary_scaffold(level: str, seed: str) -> str:
    if (level or "").lower() != "apprentice":
        return ""
    first = _pick(SCAFFOLD_STARTS, seed)
    second = _pick(SCAFFOLD_NEXT, "next-"+seed)
    return f"{first} {second}"

def enforce_mathmate_style(text: str, level: str, focus: str, auth: str, user_answer_like: bool) -> str:
    # 1) read + strip hidden tag
    tag, stripped = _extract_hidden_tag(text or "")
    t = stripped.strip()

    # 2) remove confirmations/answer-y phrases
    t = _CONFIRM_WORDS.sub(" ", t)
    t = _ANSWER_PH.sub("What makes you confident", t)

    # 3) grammar-aware rewrite BEFORE masking symbols
    t = _normalize_ops_phrasing(t)

    # 4) mask explicit arithmetic symbols & coords
    t = _EQN_BITS.sub("…", t)
    t = _COORDS.sub("that point", t)

    # 5) de-dupe micro-lesson within focus
    micro, question = _split_micro_and_question(t)
    prev_micro = last_micro(level, focus, auth)
    if micro and prev_micro and micro.strip().lower() == prev_micro.strip().lower():
        t = question or "What would you try next?"
    else:
        if micro:
            remember_micro(level, focus, auth, micro)

    # 6) Apprentice scaffolding (friendly, non-robotic)
    if (level or "").lower() == "apprentice":
        if not re.search(r"\bfirst\b|\bthen\b|\bstart\b|\bbegin\b|\bstep\b", t, re.I):
            t = (_vary_scaffold(level, focus) + " " + t).strip()

    # 7) shape & one-question rule
    t = _limit_form(t)

    # 8) Answer encouragement policy (ONLY if learner proposed a lone value)
    if user_answer_like:
        if tag == "OK":
            if not t.lstrip().startswith("✅ Try it"):
                t = "✅ Try it. " + t
        else:
            if not (t.lstrip().startswith("Mmm, let’s review the steps.") or t.lstrip().startswith("Let's check again—") or t.lstrip().startswith("Let’s step back for a sec.") ):
                t = _pick(REVIEW_PROMPTS, focus) + " " + t
    # 9) ensure one question or options
    if "?" not in t and not re.search(r"(^|\n)\s*([\-•]|A\)|1\))", t):
        t = t.rstrip(".!…") + " — what would you try next?"

    # 10) avoid identical back-to-back bot outputs
    sig = hashlib.sha1(t.encode()).hexdigest()
    if sig == last_sig(level, focus, auth):
        # lightly vary with a different reflective question (still one '?')
        base = re.sub(r"\?.*$", "", t).rstrip(".!…")
        t = f"{base}. {_pick(REFLECTIVE_QS, 'reflect-'+focus)}?"
    remember_sig(level, focus, auth, t)
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
let FOCUS = '';       // sticky anchor for the current problem
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

        # Hints for model (session + focus)
        session_line = (
            f"Session meta: level={level or 'unknown'}. "
            "If level is present, do not ask for it again; start guiding immediately."
        )
        focus_line = (
            f"Focus Anchor: {focus or '(no explicit anchor; infer from last user message/image)'} "
            "Stay on this focus and do not switch topics unless the learner clearly starts a new problem or says 'new problem'. "
            "If the learner says 'I don’t know', provide a tiny micro-lesson relevant to THIS focus and ask a smaller clarifying question."
        )

        # Style nudges
        style_line = ""
        lv = (level or "").lower()
        if lv == "apprentice":
            style_line = "Apprentice: include 1–2 scaffold lines (natural verbs; no arithmetic) before your single question."
        elif lv == "rising hero":
            style_line = "Rising Hero: one short nudge if needed, then your single question."
        elif lv == "master":
            style_line = "Master: minimal; ask one question only."

        # Detect if the user just proposed a single numeric value (e.g., "8" or "3.5")
        user_answer_like = bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", text))

        messages = [
            {"role": "system", "content": MATHMATE_PROMPT},
            {"role": "system", "content": GUIDE_RULES},
            {"role": "system", "content": HARD_CONSTRAINT},
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
        reply = enforce_mathmate_style(raw, level, focus, auth, user_answer_like)
        return jsonify(reply=reply)

    except Exception as e:
        if DEBUG:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return jsonify(error="Server error"), 500

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
