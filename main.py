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

# ---------- MATHMATE PROMPT (your rules + anchoring + guardrails) ----------
MATHMATE_PROMPT = """
🎯 MATHMATE – ACTON + KHAN ACADEMY AI GUIDE (Socratic)

ROLE
You are a Socratic math guide (not a teacher), Acton Academy style, for learners age 13+. You NEVER give answers. You help students discover them through questions, doubt, and teach-back.

HOW TO RESPOND
• ✅ Only ask QUESTIONS or offer OPTIONS. Replies are short (1–3 sentences).
• ❌ Never say or imply “correct/incorrect” or “you’re right”.
• ❌ Never explain unless asked directly.
• ✅ Nudge with: “Try it out!”, “Want to test that with the graph?”, “That might work—what makes you confident?”
• If a student proposes an answer: do NOT confirm. Ask for reasoning: “What step did you try first?”, “What made you choose that?”
  — Graphs → “Does that point match the graph?”
  — Equations → “What’s your first move?”
  — Tables → “Are the numbers consistent?”

KHAN SCREENSHOT RULES
• Check format: fraction vs decimal; which is x vs y; any graph present?
• If graph: ask for a clear point; ask which axis is which; ask what happens with y/x (unit rate).
• If thinking is right but format is off: “Great thinking—does Khan want decimal or fraction?”

CHALLENGE LEVELS
• 🐣 Apprentice — slow, define needed terms, clear step-by-step questions, patient, never give full answers.
• 🦸 Rising Hero — light support only (≤4 short sentences), ask one helpful question.
• 🧠 Master — say as little as possible; “What’s your first step?”

QUIZ STRATEGY (40/50/10)
• If total_questions is known and level ∈ {Apprentice, Rising Hero} and plan not yet announced:
  Say ONCE: “Here’s our plan 💪  40%: I’ll guide • 50%: you teach me • 10%: I’ll be quiet unless you ask.”
• Ask the learner to tell you when they start a new question so pacing matches the plan.

ALWAYS-START CHECK-IN (handled client-side; do NOT re-ask if provided)
• “How many total questions are in this exercise?”
• “Which level do you want: Apprentice, Rising Hero, or Master?”

MATH ACCURACY
• Never guess. Compute carefully. Match Khan’s requested format. Double-check which is x vs y.

DOs & DON’Ts
• ALWAYS: ask thoughtful questions; encourage reflection; match format; let the learner lead; track quiz progress; respond only with a question or options.
• NEVER: reveal the answer; say “correct”; identify exact graph coordinates; give away exact steps (e.g., “subtract 177 from 266”).

ANCHORING (critical)
• You will receive a Focus Anchor describing the current problem.
• STAY on that focus; do not switch topics unless the learner clearly starts a new problem or says “new question/new problem”.
• If the learner says “I don’t know”, keep the focus and ask a smaller clarifying question or offer 2–3 options.

STYLE
• Friendly, respectful, curious; never condescending.
• Vary emojis (max 2) from: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔.
• Do NOT write equations or name operations explicitly (avoid “add/subtract/multiply/divide”).
"""

HARD_CONSTRAINT = (
    "Hard constraint: respond ONLY with questions or short option sets; "
    "no answers, no correctness, no equations, no operation names; "
    "<= 2 sentences total and a single '?'; stay anchored to the provided focus."
)

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return "ok", 200

# ---------- UI (white, centered title, simple bubbles; in-chat onboarding; images) ----------
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
        <textarea id="msg" placeholder="Send a screenshot or paste the problem. During setup, I’ll ask total questions and your level. (Shift+Enter = newline)"></textarea>
        <div id="drop">
          <label for="fileBtn">➕ Add images (PNG/JPG) — drag & drop or click</label>
          <input id="fileBtn" type="file" accept="image/*" multiple />
          <div id="thumbs"></div>
          <small class="hint">Say “new question” when you move to the next, or “new problem” to reset focus.</small><br/>
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

let AUTH = '';
// session state (client-side to avoid model loops)
let LEVEL = '';
let TOTAL = '';
let CURRENT = 1;
let PLAN_DONE = false;
let FOCUS = '';
let onboarding = 'total'; // 'total' -> 'level' -> 'done'
let queuedImages = [];

function addBubble(who, text){
  const row = document.createElement('div');
  row.className = who === 'You' ? 'row me' : 'row bot';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.innerHTML = (text||'').replace(/</g,'&lt;');
  row.appendChild(b);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function askOnboarding(){
  if(onboarding === 'total'){
    addBubble('MathMate', "This looks like a Khan problem! How many total questions are in this exercise? 📘");
  }else if(onboarding === 'level'){
    addBubble('MathMate', "Which level do you want: 🐣 Apprentice, 🦸 Rising Hero, or 🧠 Master?");
  }else if(onboarding === 'done'){
    if(!PLAN_DONE && TOTAL && (LEVEL==='Apprentice' || LEVEL==='Rising Hero')){
      addBubble('MathMate', "Here’s our plan 💪  40%: I’ll guide • 50%: you teach me • 10%: I’ll be quiet unless you ask.");
      PLAN_DONE = true;
    }else if(LEVEL==='Master'){
      addBubble('MathMate', "Okay. You lead—what’s your first move?");
    }
  }
}

function parseInt1(text){
  const m = (text||'').match(/\\d{1,3}/);
  return m ? parseInt(m[0],10) : null;
}
function pickLevel(text){
  const t = (text||'').toLowerCase();
  if(/apprentice/.test(t)) return 'Apprentice';
  if(/rising\\s*hero/.test(t)) return 'Rising Hero';
  if(/master/.test(t)) return 'Master';
  return '';
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
  const img = document.createElement('img'); img.src = src; d.appendChild(img);
  thumbs.appendChild(d);
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
    onboarding = 'total'; askOnboarding(); msgBox.focus();
  }
};

sendBtn.onclick = async ()=>{
  const text = (msgBox.value||'').trim();
  if(!text && queuedImages.length===0) return;

  // Onboarding (client-side)
  if(onboarding !== 'done'){
    addBubble('You', text || '(image(s) only)');
    if(onboarding === 'total'){
      const n = parseInt1(text);
      if(n){ TOTAL = String(n); onboarding='level'; askOnboarding(); }
      else { addBubble('MathMate', "Type a number like 7, 10, or 15. 📘"); }
      msgBox.value=''; return;
    }
    if(onboarding === 'level'){
      const lv = pickLevel(text);
      if(lv){ LEVEL=lv; onboarding='done'; CURRENT=1; PLAN_DONE = false; askOnboarding(); }
      else { addBubble('MathMate', "Please choose: Apprentice, Rising Hero, or Master. 🙂"); }
      msgBox.value=''; return;
    }
  }

  // Normal chat
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
        text      = (p.get("message") or "").strip()
        images    = p.get("images") or []
        level     = (p.get("level") or "").strip()
        total     = (p.get("total") or "").strip()
        current   = (p.get("current") or "").strip()
        plan_done = bool(p.get("plan_done", False))
        focus     = (p.get("focus") or "").strip()

        # --- SAFE UNLOCK: return early until password matches ---
        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! Let’s set things up quick."), 200
            return jsonify(reply="🔒 Please type the access password to begin."), 200

        # Build user content (vision)
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem."}]

        # Session + focus directives
        session_line = (
            f"Session meta: level={level or 'unknown'}, total_questions={total or 'unknown'}, "
            f"current_question={current or 'unknown'}, plan_announced={'true' if plan_done else 'false'}. "
            "If level and total are known, NEVER ask for them again."
        )
        plan_rule = ""
        if level and total and not plan_done and level.lower() in ("apprentice","rising hero","risinghero"):
            plan_rule = "Announce the 40/50/10 plan ONCE now, then never mention it again."

        focus_line = (
            f"Focus Anchor: {focus or '(infer from the last learner content)'} "
            "Stay on this focus; do not switch topics unless the learner clearly starts a new problem or says 'new question/new problem'. "
            "If the learner says 'I don’t know', ask a smaller clarifying question or offer 2–3 options, but keep the same focus."
        )

        intensity_line = ""
        if (level or "").lower() == "rising hero":
            intensity_line = "Style: Rising Hero — light support (≤4 short sentences), ask one helpful question."
        elif (level or "").lower() == "master":
            intensity_line = "Style: Master — minimal; ask as little as possible."

        messages = [
            {"role": "system", "content": MATHMATE_PROMPT},
            {"role": "system", "content": focus_line},
            {"role": "system", "content": session_line},
            {"role": "system", "content": plan_rule},
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
