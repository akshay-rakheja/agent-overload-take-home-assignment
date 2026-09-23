const { chromium } = require('playwright');
const fs = require('fs');

const ARTIFACT_DIR = '/Users/akshayrakheja/.gemini/antigravity/brain/3b831073-b63e-4328-b1cc-3e5431bdbe96';

const PROMPT_1 = 'Can you check my emails for any Instagram story updates or notifications mentioning NASA or Veritasium?';
const PROMPT_2 = 'Are there any other Instagram emails mentioning SpaceX or Jeff Bezos in my feed?';

async function fetchHistories() {
  const [bRes, dRes, jRes] = await Promise.all([
    fetch('http://127.0.0.1:3000/api/chat/history?system=baseline'),
    fetch('http://127.0.0.1:3000/api/chat/history?system=enhanced_deterministic'),
    fetch('http://127.0.0.1:3000/api/chat/history?system=enhanced_jev'),
  ]);
  const [b, d, j] = await Promise.all([bRes.json(), dRes.json(), jRes.json()]);
  return {
    baselineCount: (b.messages || []).length,
    deterministicCount: (d.messages || []).length,
    jevCount: (j.messages || []).length,
    baselineLast: (b.messages || [])[(b.messages || []).length - 1],
    deterministicLast: (d.messages || [])[(d.messages || []).length - 1],
    jevLast: (j.messages || [])[(j.messages || []).length - 1],
  };
}

async function fetchInspectors() {
  const [bRes, dRes, jRes] = await Promise.all([
    fetch('http://127.0.0.1:3000/api/agents/inspector?system=baseline'),
    fetch('http://127.0.0.1:3000/api/agents/inspector?system=enhanced_deterministic'),
    fetch('http://127.0.0.1:3000/api/agents/inspector?system=enhanced_jev'),
  ]);
  return {
    baseline: await bRes.json(),
    deterministic: await dRes.json(),
    jev: await jRes.json(),
  };
}

async function waitForCondition(desc, conditionFn, timeoutMs = 120000, pollIntervalMs = 2500) {
  const start = Date.now();
  console.log(`[WAIT] Starting: ${desc}...`);
  while (Date.now() - start < timeoutMs) {
    try {
      const ok = await conditionFn();
      if (ok) {
        console.log(`[WAIT] Completed: ${desc} in ${((Date.now() - start) / 1000).toFixed(1)}s`);
        return true;
      }
    } catch (err) {
      console.warn(`[WAIT] Check error: ${err.message}`);
    }
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }
  console.warn(`[WAIT] Timed out: ${desc} after ${timeoutMs / 1000}s`);
  return false;
}

(async () => {
  console.log('[START] Launching Chromium browser...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 1100 } });
  const page = await context.newPage();

  console.log('[STEP 1] Navigating to http://127.0.0.1:3000 ...');
  await page.goto('http://127.0.0.1:3000');
  await page.waitForLoadState('networkidle');

  console.log('[STEP 2] Clearing conversation histories...');
  await fetch('http://127.0.0.1:3000/api/chat/history', { method: 'DELETE' });
  await page.reload();
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1000);

  // --- TURN 1 ---
  console.log(`[STEP 3] Entering Prompt 1: "${PROMPT_1}"`);
  const inputSelector = 'input[placeholder*="Type a prompt"]';
  await page.fill(inputSelector, PROMPT_1);
  await page.click('button[type="submit"]');

  console.log('[STEP 4] Waiting for all 3 systems to finish Turn 1 (including execution agent responses)...');
  await waitForCondition('Turn 1 complete answers in Baseline, Deterministic & Jev', async () => {
    const h = await fetchHistories();
    const bAssistants = (h.baselineCount >= 2 && h.baselineLast?.role === 'assistant' && h.baselineLast?.content?.length > 120) || h.baselineCount >= 3;
    const dAssistants = (h.deterministicCount >= 2 && h.deterministicLast?.role === 'assistant' && h.deterministicLast?.content?.length > 120) || h.deterministicCount >= 3;
    const jAssistants = (h.jevCount >= 2 && h.jevLast?.role === 'assistant' && h.jevLast?.content?.length > 120) || h.jevCount >= 3;
    console.log(`  [POLL T1] Baseline: ${h.baselineCount} msgs (${bAssistants ? '✓' : 'waiting'}), Det: ${h.deterministicCount} msgs (${dAssistants ? '✓' : 'waiting'}), Jev: ${h.jevCount} msgs (${jAssistants ? '✓' : 'waiting'})`);
    return bAssistants && dAssistants && jAssistants;
  }, 180000, 4000);

  // Wait for auto-scroll and UI settle
  await page.waitForTimeout(4000);
  const screenshotT1 = `${ARTIFACT_DIR}/tri_chat_turn_1.png`;
  await page.screenshot({ path: screenshotT1, fullPage: false });
  console.log(`[SCREENSHOT] Saved Turn 1 screenshot to ${screenshotT1}`);

  const inspT1 = await fetchInspectors();
  console.log('\n=== TURN 1 INSPECTOR TELEMETRY ===');
  console.log('Baseline:', JSON.stringify({
    roster_count: inspT1.baseline.roster_count,
    routing_action: inspT1.baseline.latest_turn?.routing_action,
    selected_agent: inspT1.baseline.latest_turn?.selected_agent_name,
  }));
  console.log('Deterministic:', JSON.stringify({
    roster_count: inspT1.deterministic.roster_count,
    routing_action: inspT1.deterministic.latest_turn?.routing_action,
    selected_agent: inspT1.deterministic.latest_turn?.selected_agent_name,
    candidates: (inspT1.deterministic.candidates || []).map(c => ({ name: c.name, score: c.score })),
  }));
  console.log('Jev:', JSON.stringify({
    roster_count: inspT1.jev.roster_count,
    routing_action: inspT1.jev.latest_turn?.routing_action,
    selected_agent: inspT1.jev.latest_turn?.selected_agent_name,
    jev_action: inspT1.jev.jev_details?.action,
    confidence: inspT1.jev.jev_details?.confidence,
    shortlist_len: (inspT1.jev.jev_details?.shortlist || []).length,
  }));

  // --- TURN 2 ---
  console.log(`\n[STEP 5] Entering Prompt 2: "${PROMPT_2}"`);
  await page.fill(inputSelector, PROMPT_2);
  await page.click('button[type="submit"]');

  console.log('[STEP 6] Waiting for all 3 systems to finish Turn 2 (including execution agent responses)...');
  await waitForCondition('Turn 2 complete answers in Baseline, Deterministic & Jev', async () => {
    const h = await fetchHistories();
    const bDone = (h.baselineCount >= 5 && h.baselineLast?.role === 'assistant' && h.baselineLast?.content?.length > 120) || h.baselineCount >= 6;
    const dDone = (h.deterministicCount >= 5 && h.deterministicLast?.role === 'assistant' && h.deterministicLast?.content?.length > 120) || h.deterministicCount >= 6;
    const jDone = (h.jevCount >= 5 && h.jevLast?.role === 'assistant' && h.jevLast?.content?.length > 120) || h.jevCount >= 6;
    console.log(`  [POLL T2] Baseline: ${h.baselineCount} msgs (${bDone ? '✓' : 'waiting'}), Det: ${h.deterministicCount} msgs (${dDone ? '✓' : 'waiting'}), Jev: ${h.jevCount} msgs (${jDone ? '✓' : 'waiting'})`);
    return bDone && dDone && jDone;
  }, 180000, 4000);

  // Wait for auto-scroll and UI settle
  await page.waitForTimeout(3000);
  const screenshotT2 = `${ARTIFACT_DIR}/tri_chat_turn_2.png`;
  await page.screenshot({ path: screenshotT2, fullPage: false });
  console.log(`[SCREENSHOT] Saved Turn 2 screenshot to ${screenshotT2}`);

  const inspT2 = await fetchInspectors();
  console.log('\n=== TURN 2 INSPECTOR TELEMETRY ===');
  console.log('Baseline:', JSON.stringify({
    roster_count: inspT2.baseline.roster_count,
    routing_action: inspT2.baseline.latest_turn?.routing_action,
    selected_agent: inspT2.baseline.latest_turn?.selected_agent_name,
  }));
  console.log('Deterministic:', JSON.stringify({
    roster_count: inspT2.deterministic.roster_count,
    routing_action: inspT2.deterministic.latest_turn?.routing_action,
    selected_agent: inspT2.deterministic.latest_turn?.selected_agent_name,
    candidates: (inspT2.deterministic.candidates || []).map(c => ({ name: c.name, score: c.score })),
  }));
  console.log('Jev:', JSON.stringify({
    roster_count: inspT2.jev.roster_count,
    routing_action: inspT2.jev.latest_turn?.routing_action,
    selected_agent: inspT2.jev.latest_turn?.selected_agent_name,
    jev_action: inspT2.jev.jev_details?.action,
    confidence: inspT2.jev.jev_details?.confidence,
    winner_margin: inspT2.jev.jev_details?.winner_margin,
    total_latency_ms: inspT2.jev.jev_details?.total_latency_ms,
    shortlist: inspT2.jev.jev_details?.shortlist,
  }));

  await browser.close();
  console.log('\n[COMPLETE] Tri-Chat live evaluation finished successfully.');
})();
