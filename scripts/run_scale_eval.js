const { chromium } = require('playwright');
const fs = require('fs');

const ARTIFACT_DIR = '/Users/akshayrakheja/.gemini/antigravity/brain/3b831073-b63e-4328-b1cc-3e5431bdbe96';

const TURNS = [
  {
    turn: 1,
    domain: 'Social / Instagram (Domain 1)',
    prompt: 'Search my emails for any Instagram notifications or creator updates mentioning NASA or Veritasium.',
    expected: 'CREATE_NEW (Social Agent)',
  },
  {
    turn: 2,
    domain: 'Finance / Receipts (Domain 2)',
    prompt: 'Check my emails for recent Uber receipts, Lyft rides, or Stripe payment invoices.',
    expected: 'CREATE_NEW (Billing Agent)',
  },
  {
    turn: 3,
    domain: 'Calendar / Meetings (Domain 3)',
    prompt: 'Find any calendar invites or meeting confirmations for next week\'s team sync.',
    expected: 'CREATE_NEW (Calendar Agent)',
  },
  {
    turn: 4,
    domain: 'Finance / Receipts (Domain 2 Reuse)',
    prompt: 'Search my emails for another Uber trip receipt or Stripe payment invoice from last week.',
    expected: 'REUSE (Billing Agent) [3 agents in roster]',
  },
  {
    turn: 5,
    domain: 'Social / Instagram (Domain 1 Return)',
    prompt: 'Check my emails for Instagram notifications and story updates from NASA.',
    expected: 'REUSE (Social Agent) [3 agents in roster]',
  },
];

async function fetchHistories() {
  const [bRes, dRes, jRes] = await Promise.all([
    fetch('http://127.0.0.1:3000/api/chat/history?system=baseline'),
    fetch('http://127.0.0.1:3000/api/chat/history?system=enhanced_deterministic'),
    fetch('http://127.0.0.1:3000/api/chat/history?system=enhanced_jev'),
  ]);
  const [b, d, j] = await Promise.all([bRes.json(), dRes.json(), jRes.json()]);
  return {
    baseline: b.messages || [],
    deterministic: d.messages || [],
    jev: j.messages || [],
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

// Helper to wait until an execution cycle settles
async function waitForTurnSettled(turnNum, timeoutMs = 120000) {
  const start = Date.now();
  console.log(`  [WAIT] Waiting for Turn ${turnNum} to settle across Baseline, Deterministic & Jev...`);
  
  // Each turn should produce an assistant reply that is not an ephemeral placeholder
  while (Date.now() - start < timeoutMs) {
    const h = await fetchHistories();
    const bLast = h.baseline[h.baseline.length - 1];
    const dLast = h.deterministic[h.deterministic.length - 1];
    const jLast = h.jev[h.jev.length - 1];

    const bDone = bLast?.role === 'assistant' && !bLast?.content?.includes('Error: OpenRouter request failed (402)');
    const dDone = dLast?.role === 'assistant' && !dLast?.content?.includes('Error: OpenRouter request failed (402)');
    const jDone = jLast?.role === 'assistant' && !jLast?.content?.includes('Error: OpenRouter request failed (402)');

    const bLen = h.baseline.length;
    const dLen = h.deterministic.length;
    const jLen = h.jev.length;

    process.stdout.write(`    [Progress] Base: ${bLen} msgs (${bDone ? '✓' : '…'}), Det: ${dLen} msgs (${dDone ? '✓' : '…'}), Jev: ${jLen} msgs (${jDone ? '✓' : '…'})\r`);

    // We consider turn done if all 3 systems have generated at least one assistant reply for this turn
    // and at least 15 seconds have elapsed since submission to let background tool execution settle
    if (bDone && dDone && jDone && (Date.now() - start >= 16000)) {
      console.log(`\n  [WAIT] Turn ${turnNum} settled in ${((Date.now() - start) / 1000).toFixed(1)}s`);
      return true;
    }
    await new Promise((r) => setTimeout(r, 2500));
  }
  console.warn(`\n  [WAIT] Turn ${turnNum} timeout reached.`);
  return false;
}

(async () => {
  console.log('================================================================');
  console.log('🚀 3-WAY MULTI-AGENT SCALING BENCHMARK');
  console.log('Baseline (:8001) vs Deterministic (:8002) vs TypeSafe Jev (:8002)');
  console.log('================================================================');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1680, height: 1100 } });
  const page = await context.newPage();

  console.log('[INIT] Opening Tri-Chat at http://127.0.0.1:3000 ...');
  await page.goto('http://127.0.0.1:3000');
  await page.waitForLoadState('networkidle');

  console.log('[INIT] Resetting all rosters and conversation logs...');
  await fetch('http://127.0.0.1:3000/api/chat/history', { method: 'DELETE' });
  await page.reload();
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1500);

  const initInsp = await fetchInspectors();
  console.log('[INIT] Initial Clean Rosters:', {
    baseline: initInsp.baseline.roster_count,
    deterministic: initInsp.deterministic.roster_count,
    jev: initInsp.jev.roster_count,
  });

  const evaluationLog = [];
  const inputSelector = 'input[placeholder*="Type a prompt"]';

  for (const t of TURNS) {
    console.log(`\n================================================================`);
    console.log(`▶ TURN ${t.turn}: ${t.domain}`);
    console.log(`  Prompt:   "${t.prompt}"`);
    console.log(`  Expected: ${t.expected}`);
    console.log(`================================================================`);

    // Enter prompt into shared input box
    await page.fill(inputSelector, t.prompt);
    await page.click('button[type="submit"]');

    // Wait for the turn to settle across all 3 systems
    await waitForTurnSettled(t.turn, 90000);

    // Allow UI auto-scroll and render settle
    await page.waitForTimeout(4000);

    // Capture visual screenshot of the 3 columns
    const screenshotPath = `${ARTIFACT_DIR}/scale_eval_turn_${t.turn}.png`;
    await page.screenshot({ path: screenshotPath, fullPage: false });
    console.log(`  [SCREENSHOT] Saved: scale_eval_turn_${t.turn}.png`);

    // Fetch telemetry from inspector
    const insp = await fetchInspectors();

    const turnReport = {
      turn: t.turn,
      domain: t.domain,
      prompt: t.prompt,
      expected: t.expected,
      baseline: {
        roster_count: insp.baseline.roster_count,
        action: insp.baseline.latest_turn?.routing_action || 'abstain',
        selected_agent: insp.baseline.latest_turn?.selected_agent_name || 'none',
        roster: (insp.baseline.roster || []).map((r) => r.name),
      },
      deterministic: {
        roster_count: insp.deterministic.roster_count,
        action: insp.deterministic.latest_turn?.routing_action || 'none',
        selected_agent: insp.deterministic.latest_turn?.selected_agent_name || 'none',
        roster: (insp.deterministic.roster || []).map((r) => r.name),
        candidates: (insp.deterministic.candidates || []).map((c) => ({ name: c.name, score: c.score })),
      },
      jev: {
        roster_count: insp.jev.roster_count,
        action: insp.jev.latest_turn?.routing_action || insp.jev.jev_details?.action || 'none',
        selected_agent: insp.jev.latest_turn?.selected_agent_name || 'none',
        confidence: insp.jev.jev_details?.confidence,
        winner_margin: insp.jev.jev_details?.winner_margin,
        map_latency_ms: insp.jev.jev_details?.map_latency_ms,
        reduce_latency_ms: insp.jev.jev_details?.reduce_latency_ms,
        total_latency_ms: insp.jev.jev_details?.total_latency_ms,
        api_calls_count: insp.jev.jev_details?.api_calls_count,
        map_scores: insp.jev.jev_details?.map_scores || [],
        roster: (insp.jev.roster || []).map((r) => r.name),
      },
    };

    console.log(`\n  --- Turn ${t.turn} Telemetry ---`);
    console.log(`  [Baseline]      Action: ${turnReport.baseline.action.toUpperCase()}, Agent: "${turnReport.baseline.selected_agent}", Roster Size: ${turnReport.baseline.roster_count}`);
    console.log(`  [Deterministic] Action: ${turnReport.deterministic.action.toUpperCase()}, Agent: "${turnReport.deterministic.selected_agent}", Roster Size: ${turnReport.deterministic.roster_count}`);
    console.log(`  [TypeSafe Jev]  Action: ${turnReport.jev.action.toUpperCase()}, Agent: "${turnReport.jev.selected_agent}", Roster Size: ${turnReport.jev.roster_count}`);
    console.log(`                  Map Latency: ${turnReport.jev.map_latency_ms}ms, Total: ${turnReport.jev.total_latency_ms}ms, Parallel API Calls: ${turnReport.jev.api_calls_count}`);
    if (turnReport.jev.map_scores.length > 0) {
      console.log(`                  Parallel Candidate Scores:`);
      for (const s of turnReport.jev.map_scores) {
        console.log(`                    • id=${s.agent_id.slice(0, 8)} composite=${s.composite_score} (aff=${s.affinity_score}, cont=${s.continuity_score}, risk=${s.risk_score})`);
      }
    }

    evaluationLog.push(turnReport);
  }

  await browser.close();

  // Save full JSON report
  const reportPath = `${ARTIFACT_DIR}/scale_eval_report.json`;
  fs.writeFileSync(reportPath, JSON.stringify(evaluationLog, null, 2), 'utf8');
  console.log(`\n[REPORT] Saved full benchmark report to ${reportPath}`);

  // Summary Table
  console.log('\n======================================================================================================');
  console.log('📊 3-WAY MULTI-AGENT SCALING BENCHMARK RESULTS');
  console.log('======================================================================================================');
  console.log('| Turn | Domain | Baseline (Action / Roster) | Deterministic (Action / Roster) | TypeSafe Jev (Action / Roster / Latency) |');
  console.log('|------|--------|----------------------------|---------------------------------|------------------------------------------|');
  for (const log of evaluationLog) {
    const baseStr = `${log.baseline.action} (Roster: ${log.baseline.roster_count})`;
    const detStr = `${log.deterministic.action} (Roster: ${log.deterministic.roster_count})`;
    const jevStr = `${log.jev.action} (Roster: ${log.jev.roster_count}) [${log.jev.total_latency_ms || 0}ms]`;
    console.log(`| T${log.turn} | ${log.domain.padEnd(30)} | ${baseStr.padEnd(26)} | ${detStr.padEnd(31)} | ${jevStr.padEnd(40)} |`);
  }
  console.log('======================================================================================================');
})();
