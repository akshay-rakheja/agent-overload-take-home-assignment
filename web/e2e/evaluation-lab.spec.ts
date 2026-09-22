import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import backend from '../tests/fixtures/backend.json';

const ready = {
  schema_version: 1, runnable: true, blockers: [], warnings: ['Controlled fabricated fixtures only.'],
  baseline: { reachable: true, revision: 'fixture-baseline', model: 'fixture-model', config_fingerprint: 'fixture-config' },
  enhanced: { reachable: true, revision: 'fixture-enhanced', model: 'fixture-model', config_fingerprint: 'fixture-config' },
  fixture_equivalence: { equivalent: true, reason: null },
  gmail_safety: { connected: true, read_only: true, reason: null },
  budget: { safe: true, remaining_usd: '1.25', reason: null },
};

async function mockLab(page: Page, options: { preflight?: unknown; run?: any; scenarios?: unknown; activeReads?: number } = {}) {
  let reads = 0;
  await page.route('**/api/lab/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/preflight')) return route.fulfill({ json: options.preflight ?? ready });
    if (path.endsWith('/scenarios')) return route.fulfill({ json: options.scenarios ?? backend.scenarios });
    if (path.endsWith('/gmail/link')) return route.fulfill({ json: { schema_version: 1, available: true, message: 'Open the local Gmail handoff window.' } });
    if (path.endsWith('/runs') && route.request().method() === 'POST') return route.fulfill({ status: 202, json: backend.handle });
    if (path.includes('/runs/')) {
      reads += 1;
      if (reads <= (options.activeReads ?? 0)) return route.fulfill({ json: { ...(options.run ?? backend.evidence_run), status: 'enhanced_running' } });
      return route.fulfill({ json: options.run ?? backend.evidence_run });
    }
    return route.fulfill({ status: 404, json: { error: 'Lab endpoint unavailable' } });
  });
}

async function axeSeriousCritical(page: Page) {
  const scan = await new AxeBuilder({ page }).analyze();
  expect(scan.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''))).toEqual([]);
}

async function noPageOverflow(page: Page) {
  expect(await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: window.innerWidth }))).toEqual(expect.objectContaining({ page: expect.any(Number), viewport: expect.any(Number) }));
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}

test('blocked preflight exposes Gmail handoff by keyboard and remains accessible', async ({ page }) => {
  await mockLab(page, { preflight: { ...ready, runnable: false, blockers: ['Gmail unsafe'], gmail_safety: { connected: false, read_only: false, reason: 'Connect the fixture mailbox.' } } });
  await page.goto('/lab');
  await expect(page.getByText('Run blocked')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run scenario' })).toBeDisabled();
  await page.getByRole('button', { name: 'Connect Gmail' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText('Open the local Gmail handoff window.')).toBeVisible();
  await axeSeriousCritical(page);
});

test('keyboard flow runs a pair, preserves inferred baseline, expands candidates and controlled evidence', async ({ page }) => {
  const repeated: any = structuredClone(backend.evidence_run);
  repeated.status = 'partial_failure';
  repeated.pairs.push({
    ...structuredClone(repeated.pairs[0]),
    scheduled: { ...repeated.pairs[0].scheduled, pair_id: '77777777-7777-4777-8777-777777777777', repetition: 2 },
    outcomes: [
      { ...structuredClone(repeated.pairs[0].outcomes[0]), status: 'failure', reason: 'Baseline fixture failed safely.', results: [] },
      structuredClone(repeated.pairs[0].outcomes[1]),
    ],
  });
  const repetitionTwoCard = structuredClone(repeated.scorecards.find((scorecard: any) => scorecard.system === 'enhanced'));
  repetitionTwoCard.pair_id = repeated.pairs[1].scheduled.pair_id;
  repetitionTwoCard.repetition = 2;
  repetitionTwoCard.turns[0].response.positive_evidence = ['Repetition 2 enhanced scorecard'];
  repeated.scorecards.push(repetitionTwoCard);
  await mockLab(page, { run: repeated });
  await page.goto('/lab');
  await page.getByLabel('Scenario').focus();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Run scenario' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Run summary' })).toBeVisible();
  await expect(page.getByText('Inferred').first()).toBeVisible();
  await expect(page.getByText('Not applicable').first()).toBeVisible();
  await expect(page.getByText('0').first()).toBeVisible();
  await axeSeriousCritical(page);
  await page.getByRole('button', { name: 'Repetition 2, 1 failure' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText('Baseline fixture failed safely.')).toBeVisible();
  await expect(page.getByText('Repetition 2 enhanced scorecard')).toBeVisible();
  await expect(page.getByText('1 failed side')).toBeVisible();
  await page.getByRole('button', { name: 'Repetition 1, 0 failures' }).focus();
  await page.keyboard.press('Enter');
  await page.getByRole('button', { name: 'Expand candidate Instagram Security Monitor' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText('exact_match 0.7')).toBeVisible();
  const controlledBody = 'Fabricated security notice SEC-7419. A sign-in was recorded at 2026-09-18 04:12 UTC from Lisbon on Pixel 10. Verification phrase: indigo-orbit.';
  await expect(page.getByText(controlledBody)).toHaveCount(0);
  await page.getByRole('button', { name: 'Show fabricated evidence' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Fabricated Gmail evidence' })).toBeVisible();
  await expect(page.getByText(controlledBody)).toBeVisible();
  await expect(page.locator('main.lab-shell')).toHaveAttribute('inert', '');
  await expect(page.getByRole('button', { name: 'Close evidence' })).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Close evidence' })).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(page.getByRole('button', { name: 'Close evidence' })).toBeFocused();
  await page.screenshot({ path: 'test-results/evaluation-lab-evidence-dialog.png' });
  await axeSeriousCritical(page);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Show fabricated evidence' })).toBeFocused();
  await expect(page.locator('main.lab-shell')).not.toHaveAttribute('inert', '');
});

test('active progress and one-side failure retain aligned successful evidence', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const partial: any = structuredClone(backend.evidence_run);
  partial.status = 'partial_failure';
  partial.pairs[0].outcomes[1] = { ...partial.pairs[0].outcomes[1], status: 'timeout', reason: 'Enhanced side exceeded its time limit.', results: [] };
  await mockLab(page, { run: partial, activeReads: 1 });
  await page.goto('/lab');
  await page.getByRole('button', { name: 'Run scenario' }).click();
  await expect(page.getByRole('status')).toContainText('Enhanced running');
  await axeSeriousCritical(page);
  await expect(page.getByRole('status')).toContainText('Partial failure', { timeout: 5000 });
  await expect(page.getByRole('region', { name: 'Baseline evidence' })).toContainText('reference: SEC-7419');
  await expect(page.getByRole('region', { name: 'Enhanced evidence' })).toContainText('Enhanced side exceeded its time limit.');
  await expect(page.getByTestId('evidence-layer')).toHaveCount(20);
  await page.screenshot({ path: 'test-results/evaluation-lab-partial-1280.png', fullPage: true });
  await axeSeriousCritical(page);
});

test('exploratory evidence stays hidden while long-history facts remain visible', async ({ page }) => {
  const exploratory: any = structuredClone(backend.evidence_run);
  const scenarios: any = structuredClone(backend.scenarios);
  exploratory.request.scenario_ids = ['optional-thousand-agent-news'];
  exploratory.pairs[0].scheduled.scenario_id = 'optional-thousand-agent-news';
  const exploratoryGmail = exploratory.pairs[0].outcomes[1].results[0].gmail_evidence.value;
  for (const event of exploratoryGmail) delete event.controlled_fixture_evidence;
  scenarios.scenarios.find((scenario: any) => scenario.scenario_id === 'optional-thousand-agent-news').track = 'natural';
  await mockLab(page, { run: exploratory, scenarios });
  await page.goto('/lab');
  await page.getByLabel('Scenario').selectOption('optional-thousand-agent-news');
  await page.getByRole('button', { name: 'Run scenario' }).click();
  await expect(page.getByText('Exploratory track')).toBeVisible();
  await expect(page.getByText('Exploratory evidence remains collapsed')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Show fabricated evidence' })).toHaveCount(0);
  await expect(page.getByText('Private appointment notes from a real mailbox.')).toHaveCount(0);
  for (const value of ['1,229,999 bytes', '10,000 entries', '9,984 omitted', 'Summary used']) await expect(page.getByText(value)).toBeVisible();
  await expect(page.getByText('Unavailable').filter({ visible: true }).first()).toBeVisible();
});

test('rejects unverified self-asserted fixture content before it can render', async ({ page }) => {
  const unsafe: any = structuredClone(backend.evidence_run);
  const gmailEvents = unsafe.pairs[0].outcomes[1].results[0].gmail_evidence.value;
  gmailEvents.at(-1).controlled_fixture_evidence = [{
    fact_id: 'not-in-manifest',
    fabricated: true,
    content: 'Private appointment notes from a real mailbox.',
  }];
  await mockLab(page, { run: unsafe });

  await page.goto('/lab');
  await page.getByRole('button', { name: 'Run scenario' }).click();

  await expect(page.getByRole('alert')).toContainText('Lab response could not be verified');
  await expect(page.getByRole('button', { name: 'Show fabricated evidence' })).toHaveCount(0);
  await expect(page.getByText('Private appointment notes from a real mailbox.')).toHaveCount(0);
});

test('multi-turn navigation keeps each turn result and exact pair scorecard aligned', async ({ page }) => {
  const multi: any = structuredClone(backend.evidence_run);
  const scenarios: any = structuredClone(backend.scenarios);
  const scenarioId = 'pronoun-receipt-follow-up';
  multi.request.scenario_ids = [scenarioId];
  multi.pairs[0].scheduled.scenario_id = scenarioId;
  for (const outcome of multi.pairs[0].outcomes) {
    const second = structuredClone(outcome.results[0]);
    second.turn_id = outcome.system === 'baseline' ? '88888888-8888-4888-8888-888888888881' : '88888888-8888-4888-8888-888888888882';
    second.final_response.value = `${outcome.system} turn two response`;
    outcome.results.push(second);
  }
  for (const scorecard of multi.scorecards) {
    scorecard.scenario_id = scenarioId;
    const second = structuredClone(scorecard.turns[0]);
    second.scenario_id = scenarioId;
    second.response.positive_evidence = [`${scorecard.system} turn two grade`];
    scorecard.turns.push(second);
  }
  await mockLab(page, { run: multi, scenarios });
  await page.goto('/lab');
  await page.getByLabel('Scenario').selectOption(scenarioId);
  await page.getByRole('button', { name: 'Run scenario' }).click();
  await expect(page.getByRole('button', { name: 'Turn 1' })).toHaveAttribute('aria-current', 'true');
  await page.getByRole('button', { name: 'Turn 2' }).click();
  await expect(page.getByRole('button', { name: 'Turn 2' })).toHaveAttribute('aria-current', 'true');
  await expect(page.getByText('baseline turn two response')).toBeVisible();
  await expect(page.getByText('enhanced turn two response')).toBeVisible();
  await expect(page.getByText('baseline turn two grade')).toBeVisible();
  await expect(page.getByText('enhanced turn two grade')).toBeVisible();
});

for (const viewport of [{ width: 1280, height: 800 }, { width: 1440, height: 900 }, { width: 1024, height: 800 }]) {
  for (const scale of [100, 1000]) test(`keeps ${scale.toLocaleString()}-agent evidence usable without page overflow at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const large = structuredClone(backend.evidence_run);
    const baseline = large.pairs[0].outcomes.find((outcome) => outcome.system === 'baseline')!.results[0];
    baseline.roster_count.value = scale;
    const selectedIndex = scale === 100 ? 77 : 777;
    const exposure = baseline.prompt_exposure.value as { exposed_names: string[]; exposed_name_count: number };
    exposure.exposed_names = Array.from({ length: scale }, (_, index) => index === selectedIndex ? `Selected ${scale}-agent identity` : `Agent ${index + 1}`);
    exposure.exposed_name_count = scale;
    baseline.selected_identity.value = { name: `Selected ${scale}-agent identity`, reason: 'one observed historical journal append' };
    await mockLab(page, { run: large });
    await page.goto('/lab');
    await page.getByRole('button', { name: 'Run scenario' }).click();
    const baselineRegion = page.getByRole('region', { name: 'Baseline evidence' });
    await expect(baselineRegion.getByText(`Roster size: ${scale.toLocaleString()}`)).toBeVisible();
    await expect(baselineRegion.getByText(`Selected: Selected ${scale}-agent identity`)).toBeVisible();
    await noPageOverflow(page);
    if (scale === 1000 && viewport.width === 1024) await page.screenshot({ path: 'test-results/evaluation-lab-1000-agents-1024.png', fullPage: true });
  });
}

test('browser fixture is accepted unchanged by the page contract', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockLab(page, { run: backend.evidence_run });
  await page.goto('/lab');
  await page.getByRole('button', { name: 'Run scenario' }).click();
  const rows = page.getByRole('table', { name: 'enhanced candidates in server order' }).locator('tbody tr');
  await expect(rows).toHaveCount(3);
  const candidateNames = await rows.evaluateAll((items) => items.map((row) => row.querySelectorAll('td')[1]?.querySelector('strong')?.textContent));
  expect(candidateNames).toEqual(['Instagram Security Monitor', 'Controlled Fixture Researcher', 'Account Security Auditor']);
  await expect(page.getByText('6073fce814d8071b361414ab24475b0bd25f504759d1517f3a762c2a8e088768')).toBeVisible();
  await axeSeriousCritical(page);
  await page.screenshot({ path: 'test-results/evaluation-lab-complete-1440.png', fullPage: true });
});
