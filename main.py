import os, re, math
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

# Pricing (USD per 1M tokens) — adjust if your plan differs
PRICE_INPUT_PER_1M        = float(os.getenv("PRICE_INPUT_PER_1M", "0.60"))
PRICE_CACHED_INPUT_PER_1M = float(os.getenv("PRICE_CACHED_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT_PER_1M       = float(os.getenv("PRICE_OUTPUT_PER_1M", "2.40"))

# Derived per-token rates
RATE_INPUT        = PRICE_INPUT_PER_1M / 1_000_000.0
RATE_CACHED_INPUT = PRICE_CACHED_INPUT_PER_1M / 1_000_000.0
RATE_OUTPUT       = PRICE_OUTPUT_PER_1M / 1_000_000.0

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- PROMPT (grounded; confirm-first for vision; unchanged logic) ----------
MATHMATE_PROMPT = r"""
🎯 MATHMATE — Teach-While-Questioning (Acton + Khan), vision-capable.

ROLE
You are a math GUIDE. You NEVER give the final numeric answer or say “correct/incorrect,” but you DO teach the method clearly while asking for the learner’s moves.

GLOBAL RULES
• Never invent or transform numbers. Use ONLY digits you can read in the learner’s message/image. If any digit is unclear, ask for a quick confirm (“For row 3 is it x=29, y=28?”).
• Do not reveal the final answer. Do not say “correct/incorrect/right/wrong.”
• You MAY name operations/formulas when explaining steps (e.g., “compute y/x for each row”), but do not print the final number.
• Stay anchored to the Focus Anchor. Avoid repetition and do not reset the conversation.
• LaTeX: always write fractions as $\\frac{y}{x}$ (never “fracyx”). Plain text fallback: (y)/(x).

VISION-GROUNDED READING (before you coach)
1) Silently read the prompt/image and extract:
   • the target constant (e.g., $k=0.9$) and the orientation (between y and x → $\\frac{y}{x}$ only — never swap),
   • the exact (x, y) pairs for the tables/options you discuss.
2) FIRST: Restate the pairs you can read and ask for confirmation. If any value is uncertain, show a “?” and ask them to type it. Do not proceed to math until they confirm.

PRIVATE CHECK (silent)
• Privately compute with the confirmed pairs only. Never print the private numbers or results.
• Use this to pick your path:
  GREEN: Looks consistent → gentle nudge to submit (“Ready to lock that in?”) or offer one quick verification choice.
  YELLOW: Missing/uncertain → ask for a tiny confirm (which row, which order, format).
  RED: Not consistent → block submission and point to the exact place to re-check (e.g., “row 2 ratio order”), without numbers.

LEVEL BEHAVIOR
• 🐣 Apprentice — Short step-by-step teaching (2–7 short sentences). State the method, then ask for one tiny action.
• 🦸 Rising Hero — ≤3 short sentences total: one hint + one guiding question/options.
• 🧠 Master — One concise guiding question only.

TEACH-WHILE-QUESTIONING
1) State the method: “Check $k=\\frac{y}{x}$ for each row.” (or the required format)
2) Do ONE micro-step together (pick a row; ask them to compute $\\frac{y}{x}$). You do not print the number.
3) After your private check, nudge/block accordingly (never say correct/incorrect).
4) Keep momentum; if you start “Table A… Table B…”, complete the current item before ending.

ANSWER-ONLY HANDLER (e.g., “A/B/C”)
• Do a PRIVATE CHECK first. If GREEN, nudge to submit or verify one row from that option. If RED, block submission and point to a specific row/ordering to re-check.

FORMAT / KHAN AWARENESS
• Match required format (fraction vs decimal). Ask to align if mismatched.

GRADE GUIDE (tone)
• K–2: ultra-simple words, one idea per sentence. 3–5: simple language + kid-friendly definitions. 6–8: standard terms; connect to unit rate. 9–12: precise terminology; justification/checks.

STYLE
Friendly, curious, never condescending. ≤2 emojis from: 🔎🧩✨💡✅🙌📘📐📊📝🎯🚀🧠📷🔧🌟🤔.
"""

HARD_CONSTRAINT = (
    "Hard constraint: do not fabricate or transform digits; if any number is uncertain, ask to confirm and WAIT for confirmation; "
    "silently compute to guide coaching but never print private calculations or the final numeric result; "
    "avoid repetition and generic resets; stay on the Focus Anchor; follow LEVEL length rules "
    "(Apprentice step-by-step; Rising Hero brief+question; Master single short question)."
)

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return "ok", 200

# ---------- UI (same, with added image auto-shrink + cost badge) ----------
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

  #composer{display:none;align-items:stretch;gap:12px}
  .inputCard{flex:1;border:1px solid var(--line);border-radius:16px;background:#fff;display:flex;flex-direction:column;overflow:hidden;transition:box-shadow .2s,border-color .2s}
  .inputCard.drag{border-color:#60a5fa;box-shadow:0 0 0 3px rgba(96,165,250,.25)}
  .inputHeader{display:flex;gap:16px;align-items:center;justify-content:flex-start;padding:10px 12px;border-bottom:1px solid var(--line);background:#f9fafb}
  .inputHeader label{display:flex;align-items:center;gap:8px;color:var(--text)}
  .inputArea{padding:10px}
  textarea{width:100%;min-height:150px;max-height:360px;resize:vertical;padding:14px;border-radius:12px;border:1px solid var(--line);outline:none;background:#fff;color:#0f172a}
  .inputFooter{border-top:1px dashed var(--line);padding:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .addBtn{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);background:#fff;cursor:pointer}
  .thumbs{display:flex;gap:8px;flex-wrap:wrap}
  .thumb{width:72px;height:72px;border:1px solid var(--line);border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .thumb img{max-width:100%;max-height:100%}
  .sendCol{display:flex;align-items:flex-end}

  /* cost badge */
  #costBar{display:none;justify-content:center;gap:10px;margin-top:8px;color:var(--muted);font-size:13px}
  #costBar span{background:#f3f4f6;border:1px solid var(--line);padding:4px 8px;border-radius:10px}
</style>

<header>
  <h1>🔒 MathMate Pro</h1>
  <div id="costBar">
    <span id="turnCost">$0.0000</span>
    <span id="dayCost">$0.0000 today</span>
  </div>
</header>

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
// ===================== COST + IMAGE SHRINK HELPERS =====================
const IMG_MAX_SIDE = 1280;
const IMG_JPEG_QUALITY = 0.82;
const HARD_LIMIT_KB = 450;   // if still bigger after shrinking, we warn (and block if strict mode on)
const STRICT_BLOCK = true;   // set to false to allow even if > HARD_LIMIT_KB (we still warn)
const ONE_IMAGE_PER_TURN = true;

// Rates (USD per 1M tokens) — should match server .env (shown to user for transparency)
const PRICE_INPUT_PER_1M = parseFloat("""" + (""" + ") + """); // dummy replaced below
</script>
<script>
// We inject server rates dynamically via data attributes for consistency
</script>
<script>
// cost UI refs
const costBar = document.getElementById('costBar');
const turnCostEl = document.getElementById('turnCost');
const dayCostEl = document.getElementById('dayCost');
let dayCost = 0;

function setRates(serverRates){
  window.RATE_INPUT_PER_TOKEN        = serverRates.rate_input;
  window.RATE_CACHED_INPUT_PER_TOKEN = serverRates.rate_cached_input;
  window.RATE_OUTPUT_PER_TOKEN       = serverRates.rate_output;
}
function fmtUSD(x){ return '$' + (Math.round(x*10000)/10000).toFixed(4); }

// image tools
async function fileToResizedDataURL(file) {
  let bmp = null;
  try { bmp = await createImageBitmap(file, { imageOrientation:'from-image' }); } catch(_){}
  if (bmp){
    const w=bmp.width, h=bmp.height;
    const scale = Math.min(1, IMG_MAX_SIDE / Math.max(w,h));
    const cw = Math.max(1, Math.round(w*scale));
    const ch = Math.max(1, Math.round(h*scale));
    const c = document.createElement('canvas'); c.width=cw; c.height=ch;
    const ctx = c.getContext('2d'); ctx.drawImage(bmp, 0,0,cw,ch); bmp.close?.();
    return c.toDataURL('image/jpeg', IMG_JPEG_QUALITY);
  }
  // fallback
  const url = URL.createObjectURL(file);
  const img = await new Promise((res,rej)=>{ const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=url; });
  const w=img.naturalWidth, h=img.naturalHeight;
  const scale = Math.min(1, IMG_MAX_SIDE / Math.max(w,h));
  const cw = Math.max(1, Math.round(w*scale));
  const ch = Math.max(1, Math.round(h*scale));
  const c=document.createElement('canvas'); c.width=cw; c.height=ch;
  const ctx=c.getContext('2d'); ctx.drawImage(img,0,0,cw,ch);
  URL.revokeObjectURL(url);
  return c.toDataURL('image/jpeg', IMG_JPEG_QUALITY);
}
function approxKBFromDataURL(dataURL){
  const b64=(dataURL.split(',')[1]||'');
  const bytes=Math.floor((b64.length*3)/4);
  return Math.round(bytes/102.4)/10;
}
async function shrinkAndMaybeBlock(file){
  const originalKB = Math.round((file.size/1024)*10)/10;
  const resized = await fileToResizedDataURL(file);
  let kb = approxKBFromDataURL(resized);
  let final = resized;

  // extra pass if still very big
  if (kb > HARD_LIMIT_KB){
    // try harder: reduce a bit more
    const img = await new Promise((res,rej)=>{ const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=resized; });
    const side = Math.round(IMG_MAX_SIDE*0.85);
    const scale = Math.min(1, side/Math.max(img.naturalWidth,img.naturalHeight));
    const cw=Math.max(1,Math.round(img.naturalWidth*scale));
    const ch=Math.max(1,Math.round(img.naturalHeight*scale));
    const c=document.createElement('canvas'); c.width=cw; c.height=ch;
    const ctx=c.getContext('2d'); ctx.drawImage(img,0,0,cw,ch);
    final = c.toDataURL('image/jpeg', Math.max(0.72, IMG_JPEG_QUALITY-0.1));
    kb = approxKBFromDataURL(final);
  }

  return { dataURL: final, originalKB, kb };
}
function showToast(msg){
  const t=document.createElement('div');
  t.textContent=msg;
  Object.assign(t.style,{
    position:'fixed', left:'50%', bottom:'16px', transform:'translateX(-50%)',
    background:'#111827', color:'#fff', padding:'10px 14px', borderRadius:'10px',
    boxShadow:'0 6px 20px rgba(0,0,0,.2)', fontSize:'13px', zIndex:9999
  });
  document.body.appendChild(t); setTimeout(()=>t.remove(), 2000);
}
</script>

<style>
/* tiny tag inside thumb for size */
.thumb .sizeTag{position:absolute;right:4px;bottom:4px;background:rgba(0,0,0,.55);color:#fff;font-size:11px;padding:2px 4px;border-radius:6px}
.thumb{position:relative}
</style>

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
const turnCostEl = document.getElementById('turnCost');
const dayCostEl = document.getElementById('dayCost');
const costBar = document.getElementById('costBar');

let AUTH=''; let LEVEL=levelSel.value; let GRADE=gradeSel.value; let CURRENT=1; let FOCUS=''; let lastBot=''; let queuedImages=[];
let HIST=[]; // rolling short history (text only)
let DAY_TOTAL_COST = 0;

function typeset(row){ if(window.MathJax?.typesetPromise){ window.MathJax.typesetPromise([row]).catch(()=>{}); } }

function addBubble(who,text){
  const content=(text||'').trim();
  if(who==='MathMate'){
    const prev=(lastBot||'').trim();
    if(prev && (content===prev || (content.length>20 && prev.length>20 && content.startsWith(prev.slice(0,Math.min(40,prev.length)))))) return;
    lastBot=content;
    HIST.push({role:'assistant', content}); HIST=HIST.slice(-6);
  }else{
    HIST.push({role:'user', content}); HIST=HIST.slice(-6);
  }
  const row=document.createElement('div'); row.className=who==='You'?'row me':'row bot';
  const bbl=document.createElement('div'); bbl.className='bubble'; bbl.innerHTML=content.replace(/</g,'&lt;');
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
  const r=await fetch('/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json','X-Auth':AUTH},
    body:JSON.stringify({ ...payload, level:LEVEL, grade:GRADE, current:CURRENT, focus:FOCUS, history:HIST })
  });
  return r.json();
}

function addThumb(src, labelText){
  const d=document.createElement('div'); d.className='thumb';
  const img=document.createElement('img'); img.src=src; d.appendChild(img);
  if(labelText){
    const tag=document.createElement('div'); tag.className='sizeTag'; tag.textContent=labelText; d.appendChild(tag);
  }
  thumbs.appendChild(d);
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

async function filesToDataURLs(files){
  for(const f of files){
    if(!f?.type?.startsWith('image/')) continue;
    if (ONE_IMAGE_PER_TURN && queuedImages.length>=1){ showToast('One image per turn, please.'); break; }
    try{
      const {dataURL, originalKB, kb} = await shrinkAndMaybeBlock(f);
      if (kb>HARD_LIMIT_KB && STRICT_BLOCK){
        showToast('Please crop tighter around the problem and try again.');
        continue; // block very large after shrinking
      }
      queuedImages.push(dataURL);
      addThumb(dataURL, `${originalKB}→${kb} KB`);
    }catch(err){
      console.warn('resize failed, sending original', err);
      const fr=new FileReader(); const p=new Promise((res,rej)=>{fr.onload=()=>res(fr.result);fr.onerror=rej;}); fr.readAsDataURL(f);
      const url=await p; queuedImages.push(url); addThumb(url);
    }
  }
}

unlockBtn.onclick=async ()=>{
  const pw=(pwdBox.value||'').trim(); if(!pw) return;
  addBubble('You','••••••••');
  const data=await post({message:pw});
  addBubble('MathMate', data.reply ?? data.error ?? '(error)');
  if((data.reply||'').startsWith('🔓')){
    AUTH=pw; unlock.style.display='none'; composer.style.display='flex'; msgBox.focus();
    // set rates from server for cost calc
    if (data.rates){ setRates(data.rates); costBar.style.display='flex'; }
  }
};

levelSel.onchange=()=>{ LEVEL=levelSel.value; };
gradeSel.onchange=()=>{ GRADE=gradeSel.value; };

sendBtn.onclick=async ()=>{
  const text=(msgBox.value||'').trim(); if(!text && queuedImages.length===0) return;
  updateFocus(text, queuedImages.length);
  addBubble('You', text || '(image(s) only)'); msgBox.value=''; sendBtn.disabled=true;
  try{
    const data=await post({message:text,images:queuedImages});
    addBubble('MathMate',(data.reply ?? data.error ?? '(error)'));
    // cost badge
    if (data.usage && data.cost){
      const c = data.cost.upper_bound_usd; // show upper bound (safe)
      DAY_TOTAL_COST += c;
      turnCostEl.textContent = fmtUSD(c);
      dayCostEl.textContent  = fmtUSD(DAY_TOTAL_COST) + ' today';
      costBar.style.display='flex';
    }
  }finally{
    sendBtn.disabled=false; queuedImages=[]; thumbs.innerHTML=''; msgBox.focus();
  }
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

        text     = str(p.get("message", "") or "").strip()
        images   = (p.get("images") or [])[:1]  # one image per turn to cap costs
        level    = str(p.get("level", "") or "").strip()
        grade    = str(p.get("grade", "") or "").strip()
        current  = str(p.get("current", "") or "").strip()
        focus    = str(p.get("focus", "") or "").strip()
        history  = p.get("history") or []  # [{role:'user'|'assistant', content:'...'}]

        # --- SAFE UNLOCK ---
        if request.headers.get("X-Auth", "") != PASSWORD:
            if text.lower() == PASSWORD.lower():
                return jsonify(
                    reply="🔓 Unlocked! Pick your grade & level, then send your problem or a photo. ✨",
                    rates={
                        "rate_input": RATE_INPUT,
                        "rate_cached_input": RATE_CACHED_INPUT,
                        "rate_output": RATE_OUTPUT,
                    },
                ), 200
            return jsonify(reply="🔒 Please type the access password to begin."), 200

        # Server-side guard: reject truly huge base64 (belt & suspenders)
        def approx_kb(data_url: str) -> int:
            try:
                b64 = (data_url.split(",", 1)[1] or "")
                return int(len(b64) * 3 / 4 / 1024)
            except Exception:
                return 0
        if any(approx_kb(u) > 800 for u in (p.get("images") or [])):
            return jsonify(reply="That picture is still pretty big. Could you crop to just the problem and resend? 📷✂️"), 200

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

        grade_line = f"GRADE={grade or 'unknown'} for tone. Use Grade Guide; simplify language for younger grades and increase rigor for older grades."
        focus_line = (
            f"Focus Anchor: {focus or '(infer from latest learner content)'} "
            "Stay on this focus; do not switch topics unless the learner clearly starts a new problem or says 'new question/new problem'."
        )

        # Vision guard: confirm pairs first when images are present
        vision_guard = ""
        if images:
            vision_guard = (
                "VISION GUARD: An image is present. Your FIRST reply must ONLY restate the (x,y) pairs you can read "
                "for any table/option you will discuss, in this compact format, without calculations: "
                "\"Read pairs → A: (x1,y1); (x2,y2) | B: ... | C: ... . Confirm Y/N?\" "
                "If any value is uncertain, use '?' and ask the learner to type it. Do not proceed to math until the learner confirms."
            )

        def add(msgs, role, content):
            if str(content or "").strip():
                msgs.append({"role": role, "content": content})

        messages = []
        add(messages, "system", MATHMATE_PROMPT)   # static → prompt-cached
        add(messages, "system", grade_line)        # small dynamic
        add(messages, "system", level_line)        # small dynamic
        add(messages, "system", focus_line)        # small dynamic
        add(messages, "system", vision_guard)      # small dynamic when images
        add(messages, "system", HARD_CONSTRAINT)   # static-ish

        # short rolling history (text-only)
        for h in history[-6:]:
            role = "assistant" if (h.get("role") == "assistant") else "user"
            content = str(h.get("content") or "").strip()
            if content:
                add(messages, role, content)

        # current turn with vision
        messages.append({"role": "user", "content": user_content})

        # --- size the reply by level so it doesn't cut off mid-list ---
        if lv == "apprentice":   max_out = 240  # (kept lean to lower output cost)
        elif lv == "rising hero":max_out = 160
        else:                    max_out = 120  # master/default

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,   # reduce randomness—no invented numbers
            frequency_penalty=0.5,
            presence_penalty=0.2,
            max_tokens=max_out,
            messages=messages,
        )

        # Cost estimate (upper bound; we can't know exact cached breakdown here)
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

        # Upper-bound cost (assume all prompt tokens at full input rate)
        cost_upper = (prompt_tokens * RATE_INPUT) + (completion_tokens * RATE_OUTPUT)

        # With-cache assumption: assume 600 of prompt tokens are the static prompt & cached after first call of the session.
        # We don't track sessions server-side, so return an optimistic scenario too.
        assumed_cached = min(600, prompt_tokens)  # tune if your prompt is longer/shorter
        cost_with_cache = ((prompt_tokens - assumed_cached) * RATE_INPUT) + (assumed_cached * RATE_CACHED_INPUT) + (completion_tokens * RATE_OUTPUT)

        return jsonify(
            reply=completion.choices[0].message.content,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
            cost={
                "upper_bound_usd": cost_upper,
                "with_cache_assumed_usd": cost_with_cache,
            },
            rates={
                "rate_input": RATE_INPUT,
                "rate_cached_input": RATE_CACHED_INPUT,
                "rate_output": RATE_OUTPUT,
            }
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        app.logger.exception("Chat crashed: %s", err)
        return jsonify(error=err if DEBUG else "Server error"), 500

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
