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

# ---------- PROMPT ----------
MATHMATE_PROMPT = r"""
🎯 MATHMATE — Teach-While-Questioning (Acton + Khan), vision-capable.

ROLE
You are a math GUIDE. You NEVER give the final numeric answer or say “correct/incorrect,” but you DO teach the method clearly while asking for the learner’s moves.

GLOBAL RULES
• Do not reveal the final answer. Do not say “correct/incorrect/right/wrong.”
• You MAY name operations and formulas when explaining steps (e.g., “compute y/x for each row”), but do not compute the final number for them.
• Stay anchored to the current problem (Focus Anchor). Do not switch topics unless the learner says “new question/new problem”.
• Avoid repetition. Do not reuse the same sentence stem twice in a row. Show the A/B/C/D operation menu at most once per question unless they ask to go back.
• Use LaTeX for math when helpful: $\\frac{y}{x}$, $20\\div 2$, $k=\\frac{y}{x}$.

LEVEL BEHAVIOR
• 🐣 Apprentice — Proactive, step-by-step teaching (2–7 short sentences allowed):
  - Explain the method in plain words, then immediately ask for a tiny action (compute, choose, or point).
  - It’s OK to name operations and the exact check (e.g., “Compute $\\frac{y}{x}$ for each pair and see if all equal 10.”).
• 🦸 Rising Hero — Brief coaching (≤3 short sentences total):
  - Give a very short method hint (≤1–2 sentences) plus one guiding question or a small options set.
• 🧠 Master — Minimal:
  - No explanations unless asked. Ask one tight question only.

TEACH-WHILE-QUESTIONING (use this flow)
1) Name the method or test FIRST in one sentence. Example: “To find a constant of proportionality $k$, check $k=\\frac{y}{x}$ for each row.”
2) Do ONE micro-step together (choose a row and ask them to compute $\\frac{y}{x}$). Do not compute it yourself.
3) If their proposal looks consistent, gently nudge to write/submit without saying it’s correct (“Ready to lock that in?”). If it looks off, block submission with a targeted check (“Before we write that, which two numbers are you comparing and in what order?”).
4) Keep momentum: after finishing a row, either (a) ask for the next row, or (b) ask to switch tables with options. Never stop mid-list.

UNSTUCK / CONTINUATION RULES
• If you start a numbered or bulleted list (e.g., “1. Table A … 2. Table B …”), you MUST complete the current item before ending your message.
• If you referenced “Table B” or “next row,” include at least the first specific prompt for it before stopping.
• If space is tight, end with a forward-moving question: “Continue with the next row or jump to Table B? A) Next row  B) Table B”.

FORMAT / KHAN AWARENESS
• Match the required format (fraction vs decimal). If their format doesn’t match, ask a format-alignment question.
• With graphs/tables, ask for a clear point or the specific ratio, and keep checking $\\frac{y}{x}$, units, and labels.

GRADE GUIDE (tone & complexity)
• K–2: ultra-simple words, one idea per sentence, concrete examples.
• 3–5: simple language plus kid-friendly definitions (“per means for each”).
• 6–8: standard terms; ask for why/how; connect to unit rate.
• 9–12: precise terminology; focus on justification/checks.

STYLE
Friendly, curious, never condescending. Use at most 2 emojis from:
🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔.
"""

HARD_CONSTRAINT = (
    "Hard constraint: never give the final numeric answer; never say ‘correct/incorrect’; "
    "you MAY name operations/formulas when explaining steps but must not compute the result; "
    "avoid repetition; stay on the Focus Anchor; follow LEVEL length rules "
    "(Apprentice longer with step-by-step; Rising Hero brief+question; Master single short question)."
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

<!-- MathJax for pretty fractions/equations -->
<script>
window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }, svg: { fontCache: 'global' } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>

<style>
  :root{--bg:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--me:#e6f0ff;--bot:#f8fafc;--accent:#111827}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial}

  /* Top header */
  header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:10;text-align:center}
  h1{margin:0;font-size:22px;letter-spacing:.2px}

  main{display:flex;justify-content:center}
  .wrap{width:100%;max-width:1400px;padding:16px}

  #chat{min-height:52vh;max-height:64vh;overflow:auto;padding:12px 4px}
  .row{display:flex;margin:10px 0}
  .bubble{max-width:78%;padding:12px 14px;border:1px solid var(--line);border-radius:16px;line-height:1.55;white-space:pre-wrap}
  .me{justify-content:flex-end}
  .me .bubble{background:var(--me)}
  .bot{justify-content:flex-start}
  .bot .bubble{background:var(--bot)}
  .sys{color:var(--muted);text-align:center;font-style:italic}

  #panel{position:sticky;bottom:0;background:var(--bg);padding:14px 0;border-top:1px solid var(--line)}
  #unlock{display:flex;gap:8px}
  input,button,select,textarea{font:inherit}
  input,select{padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--text)}
  button{padding:12px 16px;border-radius:12px;border:1px solid var(--line);background:var(--accent);color:#fff;cursor:pointer;min-width:84px}
  button:disabled{opacity:.6;cursor:not-allowed}

  /* ONE big card with grade/level + text + images */
  #composer{display:none;align-items:stretch;gap:12px}
  .inputCard{flex:1;border:1px solid var(--line);border-radius:16px;background:#fff;display:flex;flex-direction:column;overflow:hidden;transition:box-shadow .2s,border-color .2s}
  .inputCard.drag{border-color:#60a5fa;box-shadow:0 0 0 3px rgba(96,165,250,.25)}
  .inputHeader{display:flex;gap:16px;align-items:center;justify-content:flex-start;padding:10px 12px;border-bottom:1px solid var(--line);background:#f9fafb}
  .inputHeader label{display:flex;align-items:center;gap:8px;color:var(--text)}
  .inputArea{padding:10px}
  textarea{width:100%;min-height:150px;max-height:360px;resize:vertical;padding:14px;border-radius:12px;border:1px solid var(--line);outline:none;background:#fff;color:var(--text)}
  .inputFooter{border-top:1px dashed var(--line);padding:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .addBtn{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);background:#fff;cursor:pointer}
  .thumbs{display:flex;gap:8px;flex-wrap:wrap}
  .thumb{width:72px;height:72px;border:1px solid var(--line);border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  .sendCol{display:flex;align-items:flex-end}
</style>

<header><h1>🔒 MathMate Pro</h1></header>

<main><div class="wrap">
  <div id="chat"><div class="sys">Type the password to unlock.</div></div>

  <div id="panel">
    <div id="unlock">
      <input id="password" placeholder="Type the password to unlock." style="flex:1" />
      <button id="unlockBtn">Unlock</button>
    </div>

    <div id="composer">
      <div class="inputCard" id="inputCard">
        <div class="inputHeader">
          <label>Grade:
            <select id="grade">
              <option value="K">K</option>
              <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
              <option selected>6</option><option>7</option><option>8</option><option>9</option><option>10</option>
              <option>11</option><option>12</option>
            </select>
          </label>
          <label>Level:
            <select id="level">
              <option selected>Apprentice</option>
              <option>Rising Hero</option>
              <option>Master</option>
            </select>
          </label>
        </div>

        <div class="inputArea">
          <textarea id="msg" placeholder="Type here and drag/drop or paste images into this box. (Shift+Enter = newline)"></textarea>
          <input id="fileBtn" type="file" accept="image/*" multiple style="display:none" />
        </div>

        <div class="inputFooter">
          <button id="addBtn" class="addBtn" type="button">➕ Add images</button>
          <div id="thumbs" class="thumbs"></div>
        </div>
      </div>

      <div class="sendCol">
        <button id="sendBtn">Send</button>
      </div>
    </div>
  </div>
</div></main>

<script>
const chat = document.getElementById('chat');
const unlock = document.getElementById('unlock');
const composer = document.getElementById('composer');
const inputCard = document.getElementById('inputCard');
const msgBox = document.getElementById('msg');
const pwdBox = document.getElementById('password');
const unlockBtn = document.getElementById('unlockBtn');
const sendBtn = document.getElementById('sendBtn');
const fileBtn = document.getElementById('fileBtn');
const addBtn = document.getElementById('addBtn');
const thumbs = document.getElementById('thumbs');
const levelSel = document.getElementById('level');
const gradeSel = document.getElementById('grade');

let AUTH=''; let LEVEL=levelSel.value; let GRADE=gradeSel.value; let CURRENT=1; let FOCUS=''; let lastBot=''; let queuedImages=[];

function typeset(row){ if(window.MathJax?.typesetPromise){ window.MathJax.typesetPromise([row]).catch(()=>{}); } }

function addBubble(who,text){
  if(who==='MathMate'){
    const a=(text||'').trim(), b=(lastBot||'').trim();
    if(b && (a===b || (a.length>20 && b.length>20 && a.startsWith(b.slice(0,Math.min(40,b.length)))))) return;
    lastBot=a;
  }
  const row=document.createElement('div'); row.className=who==='You'?'row me':'row bot';
  const bbl=document.createElement('div'); bbl.className='bubble'; bbl.innerHTML=(text||'').replace(/</g,'&lt;');
  row.appendChild(bbl); chat.appendChild(row); chat.scrollTop=chat.scrollHeight; if(who==='MathMate') typeset(row);
}

function looksLikeProblem(t){const has=/\\d/.test(t||'');const long=(t||'').length>=16;const math=/(total|difference|sum|product|quotient|fraction|percent|rate|area|perimeter|slope|graph|table|equation|x|y)/i.test(t||'');return (has&&long)||math;}
function updateFocus(text,imgCount){
  if(/\\bnew question\\b|\\bnext question\\b/i.test(text||'')){ CURRENT=Math.max(1,CURRENT+1); return; }
  if(/\\bnew problem\\b/i.test(text||'')){ FOCUS=''; return; }
  if(imgCount>0){ FOCUS='(image problem)'; return; }
  if(looksLikeProblem(text)){ FOCUS=text.slice(0,300); }
}

async function post(payload){
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Auth':AUTH},body:JSON.stringify({...payload,level:LEVEL,grade:GRADE,current:CURRENT,focus:FOCUS})});
  return r.json();
}

function addThumb(src){ const d=document.createElement('div'); d.className='thumb'; const img=document.createElement('img'); img.src=src; d.appendChild(img); thumbs.appendChild(d); }
async function filesToDataURLs(files){
  for(const f of files){ if(!f.type.startsWith('image/')) continue;
    const fr=new FileReader(); const p=new Promise((res,rej)=>{fr.onload=()=>res(fr.result);fr.onerror=rej;}); fr.readAsDataURL(f);
    const url=await p; queuedImages.push(url); addThumb(url);
  }
}

addBtn.onclick=()=>fileBtn.click();
fileBtn.onchange=async (e)=>{ await filesToDataURLs(e.target.files); fileBtn.value=''; };

['dragenter','dragover'].forEach(ev=> inputCard.addEventListener(ev,(e)=>{ e.preventDefault(); inputCard.classList.add('drag'); }));
['dragleave','dragend','drop'].forEach(ev=> inputCard.addEventListener(ev,(e)=>{ e.preventDefault(); inputCard.classList.remove('drag'); }));
inputCard.addEventListener('drop', async (e)=>{ await filesToDataURLs(e.dataTransfer.files); });

msgBox.addEventListener('paste', async (e)=>{
  const items=e.clipboardData?.items; if(!items) return;
  const files=[]; for(const it of items){ if(it.type && it.type.startsWith('image/')) files.push(it.getAsFile()); }
  if(files.length){ e.preventDefault(); await filesToDataURLs(files); }
});

unlockBtn.onclick=async ()=>{
  const pw=(pwdBox.value||'').trim(); if(!pw) return;
  addBubble('You','••••••••');
  const data=await post({message:pw});
  addBubble('MathMate', data.reply ?? data.error ?? '(error)');
  if((data.reply||'').startsWith('🔓')){ AUTH=pw; unlock.style.display='none'; composer.style.display='flex'; msgBox.focus(); }
};

levelSel.onchange=()=>{ LEVEL=levelSel.value; };
gradeSel.onchange=()=>{ GRADE=gradeSel.value; };

sendBtn.onclick=async ()=>{
  const text=(msgBox.value||'').trim(); if(!text && queuedImages.length===0) return;
  updateFocus(text, queuedImages.length);
  addBubble('You', text || '(image(s) only)'); msgBox.value=''; sendBtn.disabled=true;
  try{ const data=await post({message:text,images:queuedImages}); addBubble('MathMate',(data.reply ?? data.error ?? '(error)')); }
  finally{ sendBtn.disabled=false; queuedImages=[]; thumbs.innerHTML=''; msgBox.focus(); }
};

msgBox.addEventListener('keydown',(e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendBtn.click(); }});
pwdBox.addEventListener('keydown',(e)=>{ if(e.key==='Enter'){ e.preventDefault(); unlockBtn.click(); }});
</script>
"""

# ---------- CHAT (vision + meta + anchor) ----------
@app.post("/chat")
def chat():
    try:
        p = request.get_json(silent=True) or {}

        text    = str(p.get("message", "") or "").strip()
        images  = (p.get("images") or [])[:4]  # safety: cap images sent to model
        level   = str(p.get("level", "") or "").strip()
        grade   = str(p.get("grade", "") or "").strip()
        current = str(p.get("current", "") or "").strip()
        focus   = str(p.get("focus", "") or "").strip()

        # --- SAFE UNLOCK ---
        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(reply="🔓 Unlocked! Pick your grade & level, then send your problem or a photo. ✨"), 200
            return jsonify(reply="🔒 Please type the access password to begin."), 200

        # Build user content (vision + text)
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        for url in images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        if not user_content:
            user_content = [{"type": "text", "text": "Please analyze the attached image problem."}]

        # Dynamic system lines
        lv = (level or "").lower()
        if lv == "apprentice":
            level_line = "LEVEL=Apprentice. You may explain proactively (2–6 short sentences) and must include a guiding question or options."
        elif lv == "rising hero":
            level_line = "LEVEL=Rising Hero. Brief coaching allowed (≤2 short sentences) plus one guiding question or options. Total 1–3 sentences."
        elif lv == "master":
            level_line = "LEVEL=Master. No explanations unless asked. One concise guiding question only."
        else:
            level_line = ""

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

        # --- size the reply by level so it doesn't cut off mid-list ---
        lv_lower = (level or "").lower()
        if lv_lower == "apprentice":
            max_out = 240
        elif lv_lower == "rising hero":
            max_out = 180
        else:
            max_out = 120  # master/default

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            frequency_penalty=0.5,
            presence_penalty=0.2,
            max_tokens=max_out,
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
