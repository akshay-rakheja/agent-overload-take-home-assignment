const { chromium } = require('playwright');
const fs = require('fs');

const ARTIFACT_DIR = '/Users/akshayrakheja/.gemini/antigravity/brain/3b831073-b63e-4328-b1cc-3e5431bdbe96';

const SCENARIOS = [
  {
    turn: 1,
    title: 'Rideshare Transit (Uber vs Lyft vs Transit)',
    prompt: 'Search my emails for my recent Uber ride receipt, trip fare, and driver tip.',
    expected_agent: 'uber_ride_receipts',
    expected_action: 'reuse',
  },
  {
    turn: 2,
    title: 'Developer Engineering (GitHub vs Linear vs Sentry)',
    prompt: 'Did anyone review or comment on my GitHub pull request or tag me in an issue today?',
    expected_agent: 'github_pull_requests',
    expected_action: 'reuse',
  },
  {
    turn: 3,
    title: 'Travel Concierge (United vs Delta vs Airbnb)',
    prompt: 'Find my United Airlines flight confirmation number and boarding pass for tomorrow\'s flight.',
    expected_agent: 'united_flight_concierge',
    expected_action: 'reuse',
  },
  {
    turn: 4,
    title: 'Subscription Services (Netflix vs Spotify vs YouTube)',
    prompt: 'Check my emails for my monthly Netflix streaming subscription receipt or plan updates.',
    expected_agent: 'netflix_subscription_manager',
    expected_action: 'reuse',
  },
  {
    turn: 5,
    title: 'Novel / Unmatched Domain (Veterinary & Pet Care)',
    prompt: 'Find my dog\'s veterinary vaccination record and pet insurance claim from Chewy.',
    expected_agent: 'CREATE_NEW',
    expected_action: 'create_new',
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

async function waitForTurnSettled(turnNum, timeoutMs = 90000) {
  const start = Date.now();
  console.log(`  [WAIT] Waiting for Scenario ${turnNum} to settle across Baseline, Deterministic & Jev...`);

  while (Date.now() - start < timeoutMs) {
    const h = await fetchHistories();
    const bLast = h.baseline[h.baseline.length - 1];
    const dLast = h.deterministic[h.deterministic.length - 1];
    const jLast = h.jev[h.jev.length - 1];

    const bDone = bLast?.role === 'assistant';
    const dDone = dLast?.role === 'assistant';
    const jDone = jLast?.role === 'assistant';

    process.stdout.write(`    [Progress] Base: ${h.baseline.length} msgs (${bDone ? '✓' : '…'}), Det: ${h.deterministic.length} msgs (${dDone ? '✓' : '…'}), Jev: ${h.jev.length} msgs (${jDone ? '✓' : '…'})\r`);

    if (bDone && dDone && jDone && (Date.now() - start >= 15000)) {
      console.log(`\n  [WAIT] Scenario ${turnNum} settled in ${((Date.now() - start) / 1000).toFixed(1)}s`);
      return true;
    }
    await new Promise((r) => setTimeout(r, 2500));
  }
  console.warn(`\n  [WAIT] Scenario ${turnNum} timeout reached.`);
  return false;
}

(async () => {
  console.log('================================================================');
  console.log('🚀 54-AGENT SCALING BENCHMARK: BASELINE vs DETERMINISTIC vs JEV');
  console.log('================================================================');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1680, height: 1100 } });
  const page = await context.newPage();

  console.log('[INIT] Opening Tri-Chat at http://127.0.0.1:3000 ...');
  await page.goto('http://127.0.0.1:3000');
  await page.waitForLoadState('networkidle');

  // Verify rosters have 50+ agents
  const initialInspectors = await fetchInspectors();
  console.log('[INIT] Verified Roster Counts (Expect >= 50):', {
    baseline: initialInspectors.baseline.roster_count,
    deterministic: initialInspectors.deterministic.roster_count,
    jev: initialInspectors.jev.roster_count,
  });

  const evaluationLog = [];
  const inputSelector = 'input[placeholder*="Type a prompt"]';

  for (const s of SCENARIOS) {
    console.log(`\n================================================================`);
    console.log(`▶ SCENARIO ${s.turn}: ${s.title}`);
    console.log(`  Prompt:   "${s.prompt}"`);
    console.log(`  Target:   Expected Agent: ${s.expected_agent} (${s.expected_action.toUpperCase()})`);
    console.log(`================================================================`);

    // Enter prompt into shared input box
    await page.fill(inputSelector, s.prompt);
    await page.click('button[type="submit"]');

    // Wait for the scenario to settle across all 3 systems
    await waitForTurnSettled(s.turn, 90000);

    // Allow UI auto-scroll and render settle
    await page.waitForTimeout(4000);

    // Capture visual screenshot of the 3 columns
    const screenshotPath = `${ARTIFACT_DIR}/eval_50_turn_${s.turn}.png`;
    await page.screenshot({ path: screenshotPath, fullPage: false });
    console.log(`  [SCREENSHOT] Saved: eval_50_turn_${s.turn}.png`);

    // Fetch telemetry from inspectors
    const insp = await fetchInspectors();

    const baselineSelected = insp.baseline.latest_turn?.selected_agent_name || 'none';
    const baselineAction = insp.baseline.latest_turn?.routing_action || 'none';
    const baselineMatched = (s.expected_agent === 'CREATE_NEW' && baselineAction === 'create_new') ||
      baselineSelected.toLowerCase().includes(s.expected_agent.replace(/_/g, ' ').toLowerCase()) ||
      baselineSelected.toLowerCase().replace(/[^a-z0-9]/g, '_').includes(s.expected_agent);

    const detSelected = insp.deterministic.latest_turn?.selected_agent_name || 'none';
    const detAction = insp.deterministic.latest_turn?.routing_action || 'none';
    const detMatched = (s.expected_agent === 'CREATE_NEW' && detAction === 'create_new') ||
      detSelected === s.expected_agent;

    const jevSelected = insp.jev.latest_turn?.selected_agent_name || 'none';
    const jevAction = insp.jev.latest_turn?.routing_action || insp.jev.jev_details?.action || 'none';
    const jevMatched = (s.expected_agent === 'CREATE_NEW' && jevAction === 'create_new') ||
      jevSelected === s.expected_agent;

    const scenarioReport = {
      turn: s.turn,
      title: s.title,
      prompt: s.prompt,
      expected: s.expected_agent,
      expected_action: s.expected_action,
      baseline: {
        roster_count: insp.baseline.roster_count,
        action: baselineAction,
        selected_agent: baselineSelected,
        matched: baselineMatched,
      },
      deterministic: {
        roster_count: insp.deterministic.roster_count,
        action: detAction,
        selected_agent: detSelected,
        matched: detMatched,
        candidates: (insp.deterministic.candidates || []).map((c) => ({ name: c.name, score: c.score })),
      },
      jev: {
        roster_count: insp.jev.roster_count,
        action: jevAction,
        selected_agent: jevSelected,
        matched: jevMatched,
        confidence: insp.jev.jev_details?.confidence,
        winner_margin: insp.jev.jev_details?.winner_margin,
        map_latency_ms: insp.jev.jev_details?.map_latency_ms,
        reduce_latency_ms: insp.jev.jev_details?.reduce_latency_ms,
        total_latency_ms: insp.jev.jev_details?.total_latency_ms,
        api_calls_count: insp.jev.jev_details?.api_calls_count,
        top_candidates: (insp.jev.jev_details?.map_scores || [])
          .sort((a, b) => b.composite_score - a.composite_score)
          .slice(0, 3)
          .map((s) => ({
            agent_id: s.agent_id,
            composite: s.composite_score,
            affinity: s.affinity_score,
            continuity: s.continuity_score,
            risk: s.risk_score,
          })),
      },
    };

    console.log(`\n  --- Scenario ${s.turn} Evaluation Telemetry ---`);
    console.log(`  [Baseline]      Action: ${baselineAction.toUpperCase()}, Selected: "${baselineSelected}", Correct: ${baselineMatched ? '✓ PASS' : '✗ FAIL'} (Roster: ${insp.baseline.roster_count})`);
    console.log(`  [Deterministic] Action: ${detAction.toUpperCase()}, Selected: "${detSelected}", Correct: ${detMatched ? '✓ PASS' : '✗ FAIL'} (Roster: ${insp.deterministic.roster_count})`);
    console.log(`  [TypeSafe Jev]  Action: ${jevAction.toUpperCase()}, Selected: "${jevSelected}", Correct: ${jevMatched ? '✓ PASS' : '✗ FAIL'} (Roster: ${insp.jev.roster_count})`);
    console.log(`                  Latency: Map=${insp.jev.jev_details?.map_latency_ms}ms, Total=${insp.jev.jev_details?.total_latency_ms}ms across 54 parallel cards`);
    if (scenarioReport.jev.top_candidates.length > 0) {
      console.log(`                  Top Jev Candidates:`);
      for (const tc of scenarioReport.jev.top_candidates) {
        console.log(`                    • id=${tc.agent_id.slice(0, 8)} composite=${tc.composite} (aff=${tc.affinity}, cont=${tc.continuity}, risk=${tc.risk})`);
      }
    }

    evaluationLog.push(scenarioReport);
  }

  await browser.close();

  // Save report
  const reportPath = `${ARTIFACT_DIR}/eval_50_report.json`;
  fs.writeFileSync(reportPath, JSON.stringify(evaluationLog, null, 2), 'utf8');
  console.log(`\n[REPORT] Saved full 54-agent evaluation report to ${reportPath}`);

  // Summary Table
  console.log('\n=================================================================================================================================');
  console.log('📊 54-AGENT SCALING BENCHMARK SUMMARY (OVERLOAD RESISTANCE & ACCURACY)');
  console.log('=================================================================================================================================');
  console.log('| Turn | Scenario | Expected | Baseline (:8001) | Deterministic (:8002) | TypeSafe Jev (:8002) [Latency] |');
  console.log('|------|----------|----------|------------------|-----------------------|--------------------------------|');
  for (const log of evaluationLog) {
    const bStr = `${log.baseline.selected_agent.slice(0, 18)} (${log.baseline.matched ? 'PASS' : 'FAIL'})`;
    const dStr = `${log.deterministic.selected_agent.slice(0, 18)} (${log.deterministic.matched ? 'PASS' : 'FAIL'})`;
    const jStr = `${log.jev.selected_agent.slice(0, 18)} (${log.jev.matched ? 'PASS' : 'FAIL'}) [${log.jev.total_latency_ms || 0}ms]`;
    console.log(`| T${log.turn} | ${log.title.padEnd(25)} | ${log.expected.padEnd(22)} | ${bStr.padEnd(16)} | ${dStr.padEnd(21)} | ${jStr.padEnd(30)} |`);
  }
  console.log('=================================================================================================================================');
})();
