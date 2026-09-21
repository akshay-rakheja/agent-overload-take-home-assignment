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


def available(value):
    return ObservedValue(availability=Availability.AVAILABLE, value=value)


def inferred(value, reason):
    return ObservedValue(availability=Availability.INFERRED, value=value, reason=reason)


def not_applicable(reason):
    return ObservedValue(availability=Availability.NOT_APPLICABLE, reason=reason)

run_id = UUID('11111111-1111-4111-8111-111111111111')
request_id = UUID('22222222-2222-4222-8222-222222222222')
turn_id = UUID('33333333-3333-4333-8333-333333333333')
now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
scenarios = load_controlled_scenarios()
scenario = scenarios[0]
schedule = build_repetition_schedule([scenario.scenario_id], repetitions=1)
baseline = SystemRunResult(
    run_id=run_id, turn_id=turn_id, system='baseline',
    revision=ObservedValue(availability=Availability.AVAILABLE, value='fixture-baseline'),
    selected_identity=ObservedValue(availability=Availability.INFERRED, value={'name': 'Fixture reader'}, reason='Observed from state delta'),
    final_response=ObservedValue(availability=Availability.AVAILABLE, value='Fabricated fixture response.'),
)

identities = [
    {'agent_id': f'00000000-0000-4000-8000-00000000000{index}', 'name': name,
     'purpose': purpose, 'status': status, 'use_count': index}
    for index, (name, purpose, status) in enumerate((
        ('Atlas', 'Account security', 'hot'),
        ('Beacon', 'Calendar planning', 'warm'),
        ('Cedar', 'Research synthesis', 'cold'),
        ('Delta', 'Draft review', 'cold'),
        ('Ember', 'Expense lookup', 'warm'),
        ('Flint', 'File organization', 'cold'),
    ), start=1)
]
baseline_evidence = SystemRunResult(
    run_id=run_id, turn_id=UUID('44444444-4444-4444-8444-444444444444'), system='baseline',
    revision=available('baseline@abc1234'), mode=inferred('full_roster', 'Historical path observed from the baseline prompt.'),
    roster_count=available(6),
    prompt_exposure=available({
        'prompt_xml_sha256': 'baseline-fixture-sha256', 'prompt_characters': 18420,
        'exposed_names': [identity['name'] for identity in identities],
        'exposed_name_count': 6, 'full_roster': True, 'identities': identities,
    }),
    candidates=not_applicable('Baseline exposes the full roster without a ranked candidate list.'),
    decision=inferred({'action': 'reuse', 'reason': 'Selected name observed after the turn.'}, 'Baseline decision inferred from state.'),
    recommendation=not_applicable('Baseline has no deterministic recommendation record.'),
    authorized_ids=not_applicable('Baseline has no dispatch authorization boundary.'),
    attempted_dispatch=ObservedValue(availability=Availability.UNAVAILABLE, reason='Baseline dispatch attempt was not emitted.'),
    accepted_dispatch=ObservedValue(availability=Availability.UNAVAILABLE, reason='Baseline dispatch acceptance was not emitted.'),
    selected_identity=inferred({'agent_id': identities[0]['agent_id'], 'name': 'Atlas'}, 'Observed from state delta.'),
    created_identity=not_applicable('No identity was created.'),
    identity_delta=available({'roster_before_count': 6, 'roster_after_count': 6, 'added_names': [], 'removed_names': []}),
    duplicates=not_applicable('Baseline has no stable-identity duplicate detector.'),
    gmail_evidence=ObservedValue(availability=Availability.UNAVAILABLE, reason='Raw baseline observation has no normalized Gmail evidence.'),
    final_response=available('Atlas found the controlled fixture and summarized the security alert.'),
    context_metrics=ObservedValue(availability=Availability.UNAVAILABLE, reason='Historical baseline has no bounded enhanced context metrics.'),
    timings=available([{'phase': 'total', 'latency_ms': 812.4, 'provider': 'fixture-provider'}]),
    usage={
        'input_tokens': available(4210), 'output_tokens': available(148), 'cached_tokens': available(0), 'total_tokens': available(4358),
        'known_input_tokens_subtotal': not_applicable('Complete subtotal not available.'),
        'known_output_tokens_subtotal': not_applicable('Complete subtotal not available.'),
        'known_cached_tokens_subtotal': not_applicable('Complete subtotal not available.'),
        'known_total_tokens_subtotal': not_applicable('Complete subtotal not available.'),
    },
    cost={'amount': available(0.0142), 'currency': available('USD'), 'known_amount_subtotal': not_applicable('Complete subtotal not available.')},
    errors=available([]), availability_metadata={'decision': Availability.INFERRED},
)
enhanced_evidence = SystemRunResult(
    run_id=run_id, turn_id=UUID('55555555-5555-4555-8555-555555555555'), system='enhanced',
    revision=available('enhanced@def5678'), mode=available('bounded_directory'), roster_count=available(6),
    prompt_exposure=available({
        'prompt_xml_sha256': 'enhanced-fixture-sha256', 'prompt_characters': 3920,
        'exposed_agent_ids': [identity['agent_id'] for identity in identities[:3]],
        'exposed_identity_count': 3, 'full_roster': False, 'identities': identities,
    }),
    candidates=available([
        {'rank': 1, 'agent_id': identities[0]['agent_id'], 'name': 'Atlas', 'status': 'hot', 'score': 0.94,
         'components': {'semantic': 0.72, 'recency': 0.12, 'use_count': 0.10}, 'reasons': ['security intent match', 'recent successful use']},
        {'rank': 2, 'agent_id': identities[2]['agent_id'], 'name': 'Cedar', 'status': 'cold', 'score': 0.61,
         'components': {'semantic': 0.55, 'recency': 0.02, 'use_count': 0.04}, 'reasons': ['research capability overlap']},
        {'rank': 3, 'agent_id': identities[3]['agent_id'], 'name': 'Delta', 'status': 'cold', 'score': 0.37,
         'components': {'semantic': 0.33, 'recency': 0.01, 'use_count': 0.03}, 'reasons': ['weak review overlap']},
    ]),
    decision=available({'action': 'reuse', 'agent_id': identities[0]['agent_id'], 'expectation': 'reuse Atlas', 'reason': 'Highest deterministic score.'}),
    recommendation=available({'action': 'reuse', 'agent_id': identities[0]['agent_id'], 'name': 'Atlas'}),
    authorized_ids=available([identities[0]['agent_id']]),
    attempted_dispatch=available({'reference_type': 'reuse', 'requested_agent_id': identities[0]['agent_id'], 'routing_action': 'reuse'}),
    accepted_dispatch=available({'status': 'accepted', 'success': True, 'selected_agent_id': identities[0]['agent_id']}),
    selected_identity=available({'agent_id': identities[0]['agent_id'], 'name': 'Atlas', 'reused': True}),
    created_identity=not_applicable('The selected identity was reused.'),
    identity_delta=available({'directory_count_before': 6, 'directory_count_after': 6, 'created_agent_ids': [], 'selected_agent_ids': [identities[0]['agent_id']]}),
    duplicates=available([]),
    gmail_evidence=available([{
        'operation_name': 'GMAIL_FETCH_EMAILS', 'stage': 'completed', 'allowed': True,
        'policy_code': 'allowed_read_only', 'query_sha256': 'controlled-query-sha256',
        'result_count': 1, 'fact_ids': ['fixture-security-001'], 'sdk_executed': True,
        'track': 'controlled', 'fabricated': True, 'content_excerpt': 'Security alert for the controlled fixture.',
        'full_fabricated_content': 'A controlled fixture reports a new sign-in and recommends reviewing account activity.',
    }]),
    final_response=available('Atlas found the controlled fixture and summarized the security alert.'),
    context_metrics=available({
        'raw_bytes': 984220, 'raw_entry_count': 10000, 'rendered_characters': 4000,
        'included_episode_count': 8, 'omitted_entry_count': 9968, 'truncated_entry_count': 24,
        'summary_used': True, 'preserved_raw_history': True, 'contamination_detected': False,
    }),
    timings=available([{'phase': 'retrieval', 'latency_ms': 19.3, 'provider': 'fixture-provider'}, {'phase': 'total', 'latency_ms': 436.8, 'provider': 'fixture-provider'}]),
    usage={
        'input_tokens': available(968), 'output_tokens': available(126), 'cached_tokens': available(0), 'total_tokens': available(1094),
        'known_input_tokens_subtotal': available(968), 'known_output_tokens_subtotal': available(126),
        'known_cached_tokens_subtotal': available(0), 'known_total_tokens_subtotal': available(1094),
    },
    cost={'amount': available(0.0048), 'currency': available('USD'), 'known_amount_subtotal': available(0.0048)},
    errors=available([]), availability_metadata={},
)
evidence_result = PairedRunResult(
    run_id=run_id, request=StartRunRequest(request_id=request_id, scenario_ids=(scenario.scenario_id,)),
    status=RunStatus.COMPLETE, generation=4, schedule=schedule,
    pairs=(PairExecutionRecord(scheduled=schedule[0], outcomes=(
        PersistedSideOutcome(system=MeasuredSystem.BASELINE, status=OutcomeStatus.SUCCESS, model_id='fixture-model-v1', results=(baseline_evidence,)),
        PersistedSideOutcome(system=MeasuredSystem.ENHANCED, status=OutcomeStatus.SUCCESS, model_id='fixture-model-v1', results=(enhanced_evidence,)),
    )),),
    scorecards=(grade_scenario_sequence(scenario, (baseline_evidence,)), grade_scenario_sequence(scenario, (enhanced_evidence,))),
    transitions=(
        RunTransition(sequence=1, status=RunStatus.QUEUED, occurred_at=now),
        RunTransition(sequence=2, status=RunStatus.COMPLETE, occurred_at=now),
    ), trace=(), created_at=now, updated_at=now,
)
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
    {'scenario_id': s.scenario_id, 'track': s.track.value, 'family': s.family, 'title': s.title,
     'repetitions': s.repetitions, 'optional': s.optional, 'budget_guarded': s.budget_guarded,
     'reset_profile': s.reset_profile.model_dump(mode='json'), 'turn_count': len(s.turns)}
    for s in scenarios
]}
print(json.dumps(redact_value({
    'scenarios': listed, 'system_result': baseline.model_dump(mode='json'),
    'handle': RunHandle(run_id=run_id, request_id=request_id, status=RunStatus.QUEUED).model_dump(mode='json'),
    'run': result.model_dump(mode='json'),
    'evidence_run': evidence_result.model_dump(mode='json'),
}), indent=2, sort_keys=True))
