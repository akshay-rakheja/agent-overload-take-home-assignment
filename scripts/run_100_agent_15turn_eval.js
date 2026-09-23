const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const ARTIFACT_DIR = process.env.ARTIFACT_DIR || path.join(__dirname, '../docs/assets');

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
    title: 'Developer Engineering (GitHub PR Reviews)',
    prompt: 'Did anyone review or comment on my GitHub pull request or tag me in an issue today?',
    expected_agent: 'github_pull_requests',
    expected_action: 'reuse',
  },
  {
    turn: 3,
    title: 'Cloud Infrastructure (AWS EC2 Compute Charges)',
    prompt: 'Check my inbox for my monthly Amazon Web Services cloud compute and EC2 billing invoice.',
    expected_agent: 'aws_cloud_billing',
    expected_action: 'reuse',
  },
  {
    turn: 4,
    title: 'Airline Concierge (United Airlines Booking)',
    prompt: 'Find my United Airlines flight confirmation number and boarding pass for tomorrow\'s flight.',
    expected_agent: 'united_flight_concierge',
    expected_action: 'reuse',
  },
  {
    turn: 5,
    title: 'Streaming Services (Netflix Subscription)',
    prompt: 'Check my emails for my monthly Netflix streaming subscription receipt or plan updates.',
    expected_agent: 'netflix_subscription_manager',
    expected_action: 'reuse',
  },
  {
    turn: 6,
    title: 'Food Delivery Distractor (Uber Eats vs Uber Rides)',
    prompt: 'Find how much I spent on my dinner food delivery order from Uber Eats last night.',
    expected_agent: 'uber_eats_receipts',
    expected_action: 'reuse',
  },
  {
    turn: 7,
    title: 'Healthcare & Doctor (Annual Physical Confirmation)',
    prompt: 'Find my upcoming doctor appointment confirmation and annual physical instructions.',
    expected_agent: 'medical_doctor_appointments',
    expected_action: 'reuse',
  },
  {
    turn: 8,
    title: 'Utilities & Power (PG&E Electric Statement)',
    prompt: 'Find my monthly electric and natural gas utility billing statement from PG&E.',
    expected_agent: 'pge_electric_utility_bills',
    expected_action: 'reuse',
  },
  {
    turn: 9,
    title: 'Banking & Financial (Chase Credit Card Statement)',
    prompt: 'Find my Chase credit card monthly electronic statement and minimum payment due.',
    expected_agent: 'chase_bank_statements',
    expected_action: 'reuse',
  },
  {
    turn: 10,
    title: 'E-Commerce Logistics (Amazon Package Delivery)',
    prompt: 'Check my emails for Amazon package shipment confirmations and tracking date.',
    expected_agent: 'amazon_delivery_tracker',
    expected_action: 'reuse',
  },
  {
    turn: 11,
    title: 'Developer Monitoring (Datadog Alert Spikes)',
    prompt: 'Search my inbox for Datadog CPU monitor alert warnings and APM error rate spikes.',
    expected_agent: 'datadog_incident_monitor',
    expected_action: 'reuse',
  },
  {
    turn: 12,
    title: 'Calendar & Team (Weekly Team Sync Invite)',
    prompt: 'Find weekly team sync calendar invite and Google Meet link for next week.',
    expected_agent: 'team_sync_scheduler',
    expected_action: 'reuse',
  },
  {
    turn: 13,
    title: 'HR & Payroll (Gusto Paycheck Deposit)',
    prompt: 'Find my latest Gusto employee direct deposit paycheck stub and salary payment.',
    expected_agent: 'gusto_payroll_stubs',
    expected_action: 'reuse',
  },
  {
    turn: 14,
    title: 'Cross-Domain Subscription (Spotify Student Plan)',
    prompt: 'Check my emails for my Spotify Premium monthly student discount subscription invoice.',
    expected_agent: 'spotify_premium_receipts',
    expected_action: 'reuse',
  },
  {
    turn: 15,
    title: 'Novel Unrepresented Domain (Veterinary & Pet Care)',
    prompt: 'Find my dog\'s veterinary rabies vaccination record and pet insurance claim from Chewy.',
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

async function waitForTurnSettled(turnNum, timeoutMs = 75000) {
  const start = Date.now();
  console.log(`  [WAIT] Waiting for Turn ${turnNum} to settle across Baseline, Deterministic & Jev...`);

  while (Date.now() - start < timeoutMs) {
    const h = await fetchHistories();
    const bLast = h.baseline[h.baseline.length - 1];
    const dLast = h.deterministic[h.deterministic.length - 1];
    const jLast = h.jev[h.jev.length - 1];

    const bDone = bLast?.role === 'assistant';
    const dDone = dLast?.role === 'assistant';
    const jDone = jLast?.role === 'assistant';

    process.stdout.write(`    [Progress] Base: ${h.baseline.length} msgs (${bDone ? '✓' : '…'}), Det: ${h.deterministic.length} msgs (${dDone ? '✓' : '…'}), Jev: ${h.jev.length} msgs (${jDone ? '✓' : '…'})\r`);

    if (bDone && dDone && jDone && (Date.now() - start >= 12000)) {
      console.log(`\n  [WAIT] Turn ${turnNum} settled in ${((Date.now() - start) / 1000).toFixed(1)}s`);
      return true;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.warn(`\n  [WAIT] Turn ${turnNum} timeout reached.`);
  return false;
}

(async () => {
  console.log('================================================================');
  console.log('🚀 100-AGENT 15-TURN BENCHMARK: BASELINE vs DETERMINISTIC vs JEV');
  console.log('================================================================\n');

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1800, height: 1100 } });

  console.log('Opening Tri-Chat UI (http://127.0.0.1:3000)...');
  await page.goto('http://127.0.0.1:3000', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  const initialInsp = await fetchInspectors();
  console.log(`Initial Rosters: Baseline=${initialInsp.baseline.roster_count}, Det=${initialInsp.deterministic.roster_count}, Jev=${initialInsp.jev.roster_count}\n`);

  const report = [];

  for (let i = 0; i < SCENARIOS.length; i++) {
    const s = SCENARIOS[i];
    console.log(`\n----------------------------------------------------------------`);
    console.log(`▶ Turn ${s.turn}/15: [${s.title}]`);
    console.log(`  Prompt: "${s.prompt}"`);
    console.log(`  Expected Target: "${s.expected_agent}" (${s.expected_action.toUpperCase()})`);

    const inputSel = 'input[placeholder*="Type a prompt"]';
    await page.fill(inputSel, s.prompt);
    await page.click('button[type="submit"]');

    await waitForTurnSettled(s.turn);
    await page.waitForTimeout(3000);

    // Capture inspector state
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

    const turnReport = {
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
        candidates: (insp.deterministic.candidates || []).slice(0, 3).map((c) => ({ name: c.name, score: c.score })),
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
      },
    };

    console.log(`  [Results Turn ${s.turn}]:`);
    console.log(`    Baseline     : Action=${baselineAction.toUpperCase()}, Selected='${baselineSelected}' -> ${baselineMatched ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`    Deterministic: Action=${detAction.toUpperCase()}, Selected='${detSelected}' -> ${detMatched ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`    TypeSafe JEV : Action=${jevAction.toUpperCase()}, Selected='${jevSelected}', Latency=${turnReport.jev.total_latency_ms}ms, Conf=${((turnReport.jev.confidence || 0) * 100).toFixed(0)}% -> ${jevMatched ? '✅ PASS' : '❌ FAIL'}`);

    report.push(turnReport);

    // Save screenshots at milestone turns
    if ([1, 5, 10, 15].includes(s.turn)) {
      const ssPath = `${ARTIFACT_DIR}/eval_100_turn_${s.turn}.png`;
      await page.screenshot({ path: ssPath, fullPage: true });
      console.log(`  📸 Saved milestone screenshot: ${ssPath}`);
    }

    await page.waitForTimeout(2000);
  }

  await browser.close();

  // Save report JSON
  const repPath = `${ARTIFACT_DIR}/eval_100_15turns_report.json`;
  fs.writeFileSync(repPath, JSON.stringify(report, null, 2), 'utf-8');
  console.log(`\n================================================================`);
  console.log(`🎉 15-Turn Benchmark Complete! Saved full report to ${repPath}`);
  console.log(`================================================================`);
})();
