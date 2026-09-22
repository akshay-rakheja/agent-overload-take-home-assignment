# Evaluation Lab Paired Run Report

## Run Metadata

- **Run ID:** `fdfa30b6-aa3c-4f8c-9b01-7de3e6ae4a29`
- **Status:** `complete`
- **Execution Mode:** `measured`
- **Model:** `openai/gpt-4.1-mini`
- **Created At:** `2026-09-22T02:53:50.122694+00:00`
- **Scenarios:** `exact-instagram-security, paraphrase-video-news, pronoun-receipt-follow-up, ambiguous-creator-clarification, novel-calendar-creation, duplicate-clipweaver-prevention, old-security-relevant, honest-no-result, hundred-agent-engagement, ten-thousand-history-anchor, instagram-security-not-engagement, similar-video-clarification, dormant-receipt-recovery, optional-thousand-agent-news`
- **Repetitions:** 1

## Paired Outcomes

| Scenario | Rep | System | Status | Mode | Candidates | Prompt Exposure | Final Response / Grade | Input Tokens | Output Tokens | Total Tokens |
|---|---:|---|---|---|---|---|---|---:|---:|---:|
| `exact-instagram-security` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: SEC-7419; timestamp: 2026-... | 4,210 | 148 | 4,358 |
| `exact-instagram-security` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: SEC-7419; timestamp: 2026-... | 968 | 126 | 1,094 |
| `paraphrase-video-news` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: NF-3207; release: Prism Cu... | 4,210 | 148 | 4,358 |
| `paraphrase-video-news` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: NF-3207; release: Prism Cu... | 968 | 126 | 1,094 |
| `pronoun-receipt-follow-up` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: VF-20481; product: Pro Ren... | 4,210 | 148 | 4,358 |
| `pronoun-receipt-follow-up` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: VF-20481; product: Pro Ren... | 4,210 | 148 | 4,358 |
| `pronoun-receipt-follow-up` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: VF-20481; product: Pro Ren... | 968 | 126 | 1,094 |
| `pronoun-receipt-follow-up` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: VF-20481; product: Pro Ren... | 968 | 126 | 1,094 |
| `ambiguous-creator-clarification` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | Which workflow should handle this? | 4,210 | 148 | 4,358 |
| `ambiguous-creator-clarification` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | available | None | Which workflow should handle this? | 968 | 126 | 1,094 |
| `novel-calendar-creation` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: MS-8820; session: Temporal... | 4,210 | 148 | 4,358 |
| `novel-calendar-creation` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | available | None | reference: MS-8820; session: Temporal... | 968 | 126 | 1,094 |
| `duplicate-clipweaver-prevention` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | Action completed. | 4,210 | 148 | 4,358 |
| `duplicate-clipweaver-prevention` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: CW-8117; total: CAD 312.40... | 4,210 | 148 | 4,358 |
| `duplicate-clipweaver-prevention` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | available | None | Action completed. | 968 | 126 | 1,094 |
| `duplicate-clipweaver-prevention` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: CW-8117; total: CAD 312.40... | 968 | 126 | 1,094 |
| `old-security-relevant` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: SEC-7419; timestamp: 2026-... | 4,210 | 148 | 4,358 |
| `old-security-relevant` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: SEC-7419; timestamp: 2026-... | 968 | 126 | 1,094 |
| `honest-no-result` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | No matching emails found. | 4,210 | 148 | 4,358 |
| `honest-no-result` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | No matching emails found. | 968 | 126 | 1,094 |
| `hundred-agent-engagement` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: ENG-2284; creator: Aurora ... | 4,210 | 148 | 4,358 |
| `hundred-agent-engagement` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: ENG-2284; creator: Aurora ... | 968 | 126 | 1,094 |
| `ten-thousand-history-anchor` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: ARC-1042; archive: Cedar C... | 4,210 | 148 | 4,358 |
| `ten-thousand-history-anchor` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: ARC-1042; archive: Cedar C... | 968 | 126 | 1,094 |
| `instagram-security-not-engagement` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: SEC-7419; timestamp: 2026-... | 4,210 | 148 | 4,358 |
| `instagram-security-not-engagement` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: SEC-7419; timestamp: 2026-... | 968 | 126 | 1,094 |
| `similar-video-clarification` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | Which workflow should handle this? | 4,210 | 148 | 4,358 |
| `similar-video-clarification` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | available | None | Which workflow should handle this? | 968 | 126 | 1,094 |
| `dormant-receipt-recovery` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: VF-20481; product: Pro Ren... | 4,210 | 148 | 4,358 |
| `dormant-receipt-recovery` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: VF-20481; product: Pro Ren... | 968 | 126 | 1,094 |
| `optional-thousand-agent-news` | 1 | Baseline | `OutcomeStatus.SUCCESS` | `full_roster` | not_applicable | 100 names (18420 chars) | reference: NF-3207; release: Prism Cu... | 4,210 | 148 | 4,358 |
| `optional-thousand-agent-news` | 1 | Enhanced | `OutcomeStatus.SUCCESS` | `bounded_directory` | 1 candidates | None | reference: NF-3207; release: Prism Cu... | 968 | 126 | 1,094 |

## Scorecards

| Scenario | Repetition | Passed | Reason / Detail |
|---|---:|---|---|
| `exact-instagram-security` | 1 | **FAIL** | 1 turns |
| `exact-instagram-security` | 1 | **FAIL** | 1 turns |
| `paraphrase-video-news` | 1 | **FAIL** | 1 turns |
| `paraphrase-video-news` | 1 | **FAIL** | 1 turns |
| `pronoun-receipt-follow-up` | 1 | **FAIL** | 2 turns |
| `pronoun-receipt-follow-up` | 1 | **FAIL** | 2 turns |
| `ambiguous-creator-clarification` | 1 | **PASS** | 1 turns |
| `ambiguous-creator-clarification` | 1 | **FAIL** | 1 turns |
| `novel-calendar-creation` | 1 | **FAIL** | 1 turns |
| `novel-calendar-creation` | 1 | **FAIL** | 1 turns |
| `duplicate-clipweaver-prevention` | 1 | **FAIL** | 2 turns |
| `duplicate-clipweaver-prevention` | 1 | **FAIL** | 2 turns |
| `old-security-relevant` | 1 | **FAIL** | 1 turns |
| `old-security-relevant` | 1 | **FAIL** | 1 turns |
| `honest-no-result` | 1 | **FAIL** | 1 turns |
| `honest-no-result` | 1 | **FAIL** | 1 turns |
| `hundred-agent-engagement` | 1 | **FAIL** | 1 turns |
| `hundred-agent-engagement` | 1 | **FAIL** | 1 turns |
| `ten-thousand-history-anchor` | 1 | **FAIL** | 1 turns |
| `ten-thousand-history-anchor` | 1 | **FAIL** | 1 turns |
| `instagram-security-not-engagement` | 1 | **FAIL** | 1 turns |
| `instagram-security-not-engagement` | 1 | **FAIL** | 1 turns |
| `similar-video-clarification` | 1 | **PASS** | 1 turns |
| `similar-video-clarification` | 1 | **FAIL** | 1 turns |
| `dormant-receipt-recovery` | 1 | **FAIL** | 1 turns |
| `dormant-receipt-recovery` | 1 | **FAIL** | 1 turns |
| `optional-thousand-agent-news` | 1 | **FAIL** | 1 turns |
| `optional-thousand-agent-news` | 1 | **FAIL** | 1 turns |
