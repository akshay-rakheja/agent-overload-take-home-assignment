"""Print sanitized, offline Python serialization for the web contract fixture.

Run from repository root: .venv/bin/python web/tests/export_lab_fixtures.py
No backend, provider, Gmail, environment configuration, or model calls are made.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.live_lab.grading import grade_scenario_sequence
from evals.live_lab.scenarios import load_controlled_scenarios
from server.services.evaluation_lab.models import Availability, ObservedValue, SystemRunResult
from server.services.evaluation_lab.orchestrator import (
    PairExecutionRecord, PairedRunResult, PersistedSideOutcome, RunHandle,
    RunStatus, StartRunRequest, RunTransition,
)
from server.services.evaluation_lab.repetitions import MeasuredSystem, OutcomeStatus, build_repetition_schedule
from server.services.evaluation_lab.redaction import redact_value

run_id = UUID('11111111-1111-4111-8111-111111111111')
request_id = UUID('22222222-2222-4222-8222-222222222222')
turn_id = UUID('33333333-3333-4333-8333-333333333333')
now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
scenarios = load_controlled_scenarios()
scenario = scenarios[0]
baseline = SystemRunResult(
    run_id=run_id, turn_id=turn_id, system='baseline',
    revision=ObservedValue(availability=Availability.AVAILABLE, value='fixture-baseline'),
    selected_identity=ObservedValue(availability=Availability.INFERRED, value={'name': 'Fixture reader'}, reason='Observed from state delta'),
    final_response=ObservedValue(availability=Availability.AVAILABLE, value='Fabricated fixture response.'),
)
schedule = build_repetition_schedule([scenario.scenario_id], repetitions=1)
result = PairedRunResult(
    run_id=run_id, request=StartRunRequest(request_id=request_id, scenario_ids=(scenario.scenario_id,)),
    status=RunStatus.PARTIAL_FAILURE, generation=3, schedule=schedule,
    pairs=(PairExecutionRecord(scheduled=schedule[0], outcomes=(
        PersistedSideOutcome(system=MeasuredSystem.BASELINE, status=OutcomeStatus.SUCCESS, model_id='fixture-model', results=(baseline,)),
        PersistedSideOutcome(system=MeasuredSystem.ENHANCED, status=OutcomeStatus.TIMEOUT, reason='Enhanced side exceeded its time limit.'),
    )),),
    scorecards=(grade_scenario_sequence(scenario, (baseline,)),),
    transitions=(RunTransition(sequence=1, status=RunStatus.PARTIAL_FAILURE, occurred_at=now),),
    trace=(), created_at=now, updated_at=now,
)
listed = {'schema_version': 1, 'scenario_count': len(scenarios), 'scenarios': [
    {'scenario_id': s.scenario_id, 'family': s.family, 'title': s.title,
     'repetitions': s.repetitions, 'optional': s.optional, 'budget_guarded': s.budget_guarded,
     'reset_profile': s.reset_profile.model_dump(mode='json'), 'turn_count': len(s.turns)}
    for s in scenarios
]}
print(json.dumps(redact_value({
    'scenarios': listed, 'system_result': baseline.model_dump(mode='json'),
    'handle': RunHandle(run_id=run_id, request_id=request_id, status=RunStatus.QUEUED).model_dump(mode='json'),
    'run': result.model_dump(mode='json'),
}), indent=2, sort_keys=True))
