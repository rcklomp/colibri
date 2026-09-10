// ui_probe.mjs — drive the owner's Open WebUI in a real browser, the way a person does.
//
// Written after P7b (2026-09-09): every gate on this track imitated the client, and both
// things that broke were places where the real front end differs from the imitation. This
// one IS the client: a Chromium page at http://<rig>:3000, a new chat, a typed question, and
// the tokens as they appear on screen. It measures what the person waits for -- the first
// visible token -- not what the server logs.
//
//   OWUI_TOKEN=<jwt> node ui_probe.mjs --url http://rome.local:3000 [--question "..."]
//        [--timeout 900] [--headed] [--shot out.png] [--keep-chat]
//        [--follow-up "..."] [--follow-ups N] [--nonce TEXT]
// Prints one machine-readable line:
//   RESULT ok=<0|1> first_token_s=<s> done_s=<s> chars=<n> chat=<id|-> error=<...>
// plus one `TURN <n> first_token_s=.. done_s=..` line per turn.
//
// P9: --follow-ups N sends N follow-up turns in the SAME chat, which is the
// production shape the ledger's invariant has to hold in -- ten turns of a real
// browser conversation with zero MISMATCH in the gateway's log. Each follow-up
// carries the turn number and the nonce, because a check whose question is fixed
// makes a second run send an identical turn that the engine cannot reuse BY
// DESIGN (kv_prefix_reuse needs at least one new token), and that reads as a
// regression which is really the harness repeating itself.
//
// The token is read from the environment and never printed. Mint it on the rig
// (tools/hot-expert/accept_live.sh does) and pass it in; the browser then signs in as the
// owner without a password ever leaving the box.
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';

// playwright-core lives outside the repo (no node_modules in git). accept_ui.sh installs it
// into ~/.cache/colibri-ui; a plain `npm i playwright-core` anywhere on NODE_PATH also works.
const require_ = createRequire(import.meta.url);
const roots = [process.env.COLIBRI_UI_MODULES, `${homedir()}/.cache/colibri-ui/node_modules`,
               `${homedir()}/node_modules`].filter(Boolean);
let chromium = null;
for (const r of roots) {
  const p = `${r}/playwright-core`;
  if (existsSync(p)) { chromium = require_(p).chromium; break; }
}
if (!chromium) { try { chromium = require_('playwright-core').chromium; } catch {} }
if (!chromium) {
  console.log('RESULT ok=0 error=playwright-core_not_found_run_accept_ui.sh');
  process.exit(2);
}

const arg = (name, dflt = null) => {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return dflt;
  const next = process.argv[i + 1];
  return (!next || next.startsWith('--')) ? true : next;
};
const url = String(arg('url', 'http://rome.local:3000')).replace(/\/$/, '');
let question = String(arg('question', 'Which day comes after Tuesday? Answer in one word.'));
const textFile = arg('text-file', null);
if (textFile) question = (await import('node:fs')).readFileSync(String(textFile), 'utf8');
const followUp = arg('follow-up', null);
const followUps = Number(arg('follow-ups', followUp ? 1 : 0)) || 0;
const nonce = String(arg('nonce', String(Date.now()).slice(-6)));
const timeoutMs = Number(arg('timeout', 900)) * 1000;
const shot = arg('shot', null);
const token = process.env.OWUI_TOKEN;
if (!token) { console.log('RESULT ok=0 error=no_OWUI_TOKEN'); process.exit(2); }

const out = { ok: 0, first_token_s: '-', done_s: '-', chars: 0, chat: '-', error: '-', reply: '', cleaned: '-', browser: 'bundled' };
const done = (code) => {
  console.log(`RESULT ok=${out.ok} first_token_s=${out.first_token_s} done_s=${out.done_s} ` +
              `follow_first_s=${out.follow_first_s ?? '-'} follow_done_s=${out.follow_done_s ?? '-'} ` +
              `turns=${out.turns ?? 1} chars=${out.chars} chat=${out.chat} cleaned=${out.cleaned} browser=${out.browser} error=${out.error}` +
              (out.reply ? ` reply="${out.reply}"` : ''));
  process.exit(code);
};

// Prefer Playwright's own build; fall back to the Mac's installed Google Chrome when the
// driver and the cached browser revision disagree (npm gives the newest driver, the cache
// holds whatever an older one downloaded). A gate that needs a download to run is a gate
// that stops working on a bad network day.
const headless = arg('headed') !== true;
let browser = null;
try {
  browser = await chromium.launch({ headless });
} catch (e) {
  if (!/Executable doesn't exist/i.test(String(e.message || e))) throw e;
  try {
    browser = await chromium.launch({ headless, channel: 'chrome' });
    out.browser = 'system-chrome';   // NOTE: ~10 s slower to first paint than the bundled
                                      // headless shell on the same turn -- not comparable
  } catch (e2) {
    out.error = 'no_browser:' + String(e2.message || e2).split('\n')[0].slice(0, 80).replace(/\s+/g, '_');
    done(2);
  }
}
const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const host = new URL(url).hostname;
await context.addCookies([{ name: 'token', value: token, domain: host, path: '/' }]);
const page = await context.newPage();
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(String(e).slice(0, 120)));

try {
  // The front end keeps its session in localStorage as well as the cookie.
  await page.addInitScript((t) => { try { localStorage.setItem('token', t); } catch {} }, token);
  await page.goto(url + '/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => {});

  // The composer: a ProseMirror contenteditable in 0.11.x, a textarea in older builds.
  const box = page.locator('#chat-input, textarea#chat-textarea, [contenteditable="true"]').first();
  await box.waitFor({ state: 'visible', timeout: 60000 });
  await box.click();
  // insertText, not per-character typing: a 4 000-token paste would otherwise spend two
  // minutes in the keyboard before the request is even sent, which is not what we measure.
  if (question.length > 400) await page.keyboard.insertText(question);
  else await box.type(question, { delay: 8 });

  // The turns already on screen, so the new assistant bubble can be told apart.
  const before = await page.evaluate(() => [...document.querySelectorAll('[id^="message-"]')].map((n) => n.id));

  // Everything after this is what the person experiences.
  const t0 = Date.now();
  await page.keyboard.press('Enter');

  // First visible token in the ASSISTANT bubble. Open WebUI renders every turn as
  // `[id^="message-"]`; the person's own turn carries `user-message`, so anything else with
  // text is the model. (The first draft timed the user's own bubble and read 0.05 s — the
  // same mistake as the gates this file exists to replace: measuring the wrong request.)
  const firstAt = await page.waitForFunction((known) => {
    const isAssistant = (n) => n.id !== 'message-input-container' &&
      !(n.className || '').toString().includes('user-message');
    for (const n of document.querySelectorAll('[id^="message-"]')) {
      if (!isAssistant(n) || known.includes(n.id)) continue;
      const body = n.querySelector('.markdown-prose, .prose');
      if (body && (body.innerText || '').trim().length > 0) return Date.now();
    }
    return false;
  }, before, { timeout: timeoutMs, polling: 250 }).then((h) => h.jsonValue());
  out.first_token_s = ((firstAt - t0) / 1000).toFixed(2);

  // Done: the stop button is gone and the text has stopped growing.
  await page.waitForFunction(() => {
    const stop = document.querySelector('#stop-response-button, button[aria-label*="Stop" i]');
    if (stop && stop.offsetParent !== null) return false;
    const nodes = [...document.querySelectorAll('[id^="message-"]')].filter((n) =>
      n.id !== 'message-input-container' && !(n.className || '').toString().includes('user-message'));
    const last = nodes.pop();
    if (!last) return false;
    const body = last.querySelector('.markdown-prose, .prose');
    const len = body ? (body.innerText || '').trim().length : 0;
    if (!len) return false;
    if (window.__probeLen === len) return true;      // unchanged since the last poll
    window.__probeLen = len; return false;
  }, null, { timeout: timeoutMs, polling: 800 }).catch(() => {});
  out.done_s = ((Date.now() - t0) / 1000).toFixed(2);

  const reply = await page.evaluate((known) => {
    const nodes = [...document.querySelectorAll('[id^="message-"]')].filter((n) =>
      n.id !== 'message-input-container' && !(n.className || '').toString().includes('user-message')
      && !known.includes(n.id));
    const last = nodes.pop();
    const body = last && last.querySelector('.markdown-prose, .prose');
    return body ? body.innerText.trim() : '';
  }, before);
  out.chars = reply.length;
  out.reply = reply.slice(0, 60).replace(/\s+/g, ' ');
  const m = page.url().match(/\/c\/([0-9a-f-]{36})/);
  out.chat = m ? m[1] : '-';
  out.ok = out.chars > 0 ? 1 : 0;
  if (!out.ok) out.error = 'empty_reply';

  console.log(`TURN 1 first_token_s=${out.first_token_s} done_s=${out.done_s} chars=${out.chars}`);

  // Further turns in the SAME chat: what every reply after the first costs.
  // The first of them keeps the historic follow_first_s / follow_done_s names,
  // so ui_matrix.sh and accept_ui.sh read exactly what they read before.
  const askedQuestions = [];
  for (let turn = 2; turn <= followUps + 1; turn++) {
    const text = followUps > 1
      ? `Turn ${turn} [${nonce}]: name one ${['colour', 'animal', 'city', 'metal', 'river',
          'planet', 'fruit', 'instrument', 'language', 'mountain'][(turn - 2) % 10]}. One word.`
      : String(followUp);
    askedQuestions.push(text);
    const beforeN = await page.evaluate(() => [...document.querySelectorAll('[id^="message-"]')].map((n) => n.id));
    const boxN = page.locator('#chat-input, textarea#chat-textarea, [contenteditable="true"]').first();
    await boxN.click();
    await boxN.type(text, { delay: 8 });
    const tN = Date.now();
    await page.keyboard.press('Enter');
    const firstN = await page.waitForFunction((known) => {
      const isAssistant = (n) => n.id !== 'message-input-container' &&
        !(n.className || '').toString().includes('user-message');
      for (const n of document.querySelectorAll('[id^="message-"]')) {
        if (!isAssistant(n) || known.includes(n.id)) continue;
        const body = n.querySelector('.markdown-prose, .prose');
        if (body && (body.innerText || '').trim().length > 0) return Date.now();
      }
      return false;
    }, beforeN, { timeout: timeoutMs, polling: 250 }).then((h) => h.jsonValue());
    // Done: the stop button is gone AND the last bubble stopped growing. Waiting
    // only for the button lets the next turn's Enter land while the previous
    // reply is still streaming, which is not a conversation a person could have.
    await page.waitForFunction(() => {
      const stop = document.querySelector('#stop-response-button, button[aria-label*="Stop" i]');
      if (stop && stop.offsetParent !== null) return false;
      const nodes = [...document.querySelectorAll('[id^="message-"]')].filter((n) =>
        n.id !== 'message-input-container' && !(n.className || '').toString().includes('user-message'));
      const last = nodes.pop();
      const body = last && last.querySelector('.markdown-prose, .prose');
      const len = body ? (body.innerText || '').trim().length : 0;
      if (!len) return false;
      if (window.__probeLenN === len) return true;
      window.__probeLenN = len; return false;
    }, null, { timeout: timeoutMs, polling: 800 }).catch(() => {});
    await page.evaluate(() => { window.__probeLenN = -1; });
    const firstS = ((firstN - tN) / 1000).toFixed(2);
    const doneS = ((Date.now() - tN) / 1000).toFixed(2);
    if (turn === 2) { out.follow_first_s = firstS; out.follow_done_s = doneS; }
    out.turns = turn;
    console.log(`TURN ${turn} first_token_s=${firstS} done_s=${doneS}`);
  }
  void askedQuestions;

  // Leave no trace: the probe's chat is deleted unless --keep-chat.
  if (out.chat !== '-' && arg('keep-chat') !== true) {
    const gone = await page.evaluate(async ([base, id, t]) => {
      const r = await fetch(`${base}/api/v1/chats/${id}`, { method: 'DELETE', headers: { Authorization: `Bearer ${t}` } });
      return r.status;
    }, [url, out.chat, token]);
    out.cleaned = gone === 200 ? 1 : gone;
  }
} catch (e) {
  out.error = String(e.message || e).split('\n')[0].slice(0, 140).replace(/\s+/g, '_');
} finally {
  if (shot) { try { await page.screenshot({ path: String(shot), fullPage: false }); } catch {} }
  if (pageErrors.length && out.error === '-') out.error = 'pageerror:' + pageErrors[0].replace(/\s+/g, '_');
  await browser.close().catch(() => {});
}
done(out.ok ? 0 : 1);
