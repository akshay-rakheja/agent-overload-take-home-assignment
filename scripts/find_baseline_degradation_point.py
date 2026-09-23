#!/usr/bin/env python3
"""Empirical degradation sweep for Baseline OpenPoke roster size N.

Tests baseline execution agent selection across increasing roster sizes N in [10, 25, 50, 75, 100, 125, 150]
using a 2-step interaction loop mimicking Baseline's exact runtime.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

repo_dir = Path(__file__).resolve().parent.parent
default_base = repo_dir.parent / "openpoke-evaluation-baseline"
base_path = Path(os.getenv("BASELINE_PATH", default_base))
if not base_path.exists():
    base_path = Path("/Users/akshayrakheja/Documents/general-magic-take-home/openpoke-evaluation-baseline")
sys.path.insert(0, str(base_path))

from dotenv import load_dotenv
load_dotenv(base_path / ".env")

from server.agents.interaction_agent.agent import build_system_prompt
from server.agents.interaction_agent.tools import TOOL_SCHEMAS
from server.openrouter_client import request_chat_completion

# 150 unique realistic execution agents across diverse domains
MASTER_AGENT_POOL = [
    # Rideshare & Transit (0..4)
    "uber_ride_receipts", "lyft_transit_receipts", "caltrain_transit_tracker", "waymo_autonomous_trips", "lime_scooter_receipts",
    # Food Delivery & Dining (5..9)
    "uber_eats_receipts", "doordash_order_tracker", "instacart_grocery_receipts", "opentable_dining_reservations", "seamless_grubhub_delivery",
    # Cloud Billing & Invoicing (10..15)
    "aws_cloud_billing", "gcp_cloud_billing", "azure_cloud_billing", "stripe_payout_notifier", "paypal_payment_receipts", "vercel_usage_invoices",
    # Dev & Engineering (16..21)
    "github_pull_requests", "gitlab_pipeline_monitor", "linear_issue_tracker", "jira_sprint_notifier", "sentry_error_alerts", "datadog_incident_monitor",
    # Travel & Lodging (22..27)
    "united_flight_concierge", "delta_flight_tracker", "airbnb_reservation_assistant", "marriott_hotel_receipts", "rental_car_receipts", "alaska_airlines_travel",
    # E-Commerce & Logistics (28..33)
    "amazon_delivery_tracker", "ups_package_tracker", "fedex_delivery_monitor", "apple_store_receipts", "bestbuy_order_pickup", "target_circle_receipts",
    # Calendar & Meetings (34..38)
    "team_sync_scheduler", "manager_1on1_scheduler", "interview_scheduling_assistant", "zoom_meeting_recordings", "calendly_booking_notifications",
    # Subscriptions & Streaming (39..44)
    "netflix_subscription_manager", "spotify_premium_receipts", "youtube_premium_receipts", "nytimes_subscription_receipts", "chatgpt_plus_receipts", "adobe_creative_cloud",
    # Healthcare & Wellness (45..49)
    "medical_doctor_appointments", "dental_cleaning_scheduler", "gym_fitness_membership", "pharmacy_prescription_refills", "optometry_eye_exam_tracker",
    # Utilities & Telecom (50..54)
    "pge_electric_utility_bills", "coned_gas_statements", "verizon_wireless_bills", "comcast_xfinity_monthly_bill", "tmobile_family_plan",
    # Banking & Investments (55..59)
    "chase_bank_statements", "bank_of_america_alerts", "fidelity_401k_contributions", "vanguard_index_fund_dividends", "robinhood_stock_trade_confirms",
    # Social & Creator (60..65)
    "instagram_story_monitor", "twitter_mention_tracker", "linkedin_networking_agent", "youtube_creator_notifier", "tiktok_creator_alert", "reddit_digest_scanner",
    # Smart Home & Security (66..69)
    "ring_doorbell_motion_alerts", "nest_thermostat_energy_report", "arlo_security_camera_events", "philips_hue_lighting_schedules",
    # Education & Learning (70..73)
    "coursera_course_certificates", "duolingo_streak_reminders", "udemy_course_receipts", "submittable_literary_submissions",
    # Work & HR (74..77)
    "gusto_payroll_stubs", "adp_tax_w2_statements", "greenhouse_applicant_reviews", "workday_timeoff_requests",
    # Real Estate & Housing (78..81)
    "zillow_home_listings", "redfin_price_drop_alerts", "apartments_com_tour_requests", "hoa_monthly_assessment_notices",
    # Auto & Insurance (82..85)
    "geico_auto_insurance_policy", "progressive_vehicle_id_cards", "dmv_registration_renewal_alerts", "jiffylube_oil_change_reminders",
    # Entertainment & Tickets (86..89)
    "ticketmaster_concert_tickets", "stubhub_event_passes", "amc_movie_theater_tickets", "fandango_movie_tickets",
    # Gaming & Media (90..94)
    "steam_game_purchase_receipts", "playstation_network_renewals", "nintendo_eshop_receipts", "patreon_membership_pledges", "audible_audiobook_credits",
    # Reading & News (95..99)
    "substack_newsletter_curator", "morning_brew_digest", "medium_daily_digest", "hackernews_comment_replies", "kindle_unlimited_borrow_alerts",
    # Productivity Tools (100..104)
    "evernote_monthly_backup", "notion_workspace_invites", "slack_direct_message_mentions", "discord_server_announcements", "asana_project_milestones",
    # Cloud Storage (105..108)
    "dropbox_storage_quota_alerts", "box_enterprise_collaboration", "google_drive_shared_folders", "icloud_storage_upgrade_receipts",
    # Travel aggregators (109..112)
    "expedia_hotel_package_deals", "kayak_price_forecast_alerts", "hopper_flight_watchers", "booking_com_rewards_status",
    # Big Retail & Hardware (113..116)
    "costco_wholesale_membership_renewal", "sam_club_pickup_orders", "ikea_furniture_delivery_window", "homedepot_tool_rental_receipts",
    # Pets & Hobbies (117..120)
    "chewy_pet_food_orders", "barkbox_monthly_toy_delivery", "petco_grooming_appointments", "rei_outdoor_gear_receipts",
    # Extra unique agents (121..149)
    "spotify_podcasts_digest", "tesla_supercharging_receipts", "sonos_speaker_system_updates", "bose_headphones_warranty",
    "dyson_vacuum_filter_replacements", "hellofresh_meal_box_delivery", "blueapron_weekly_recipe_shipment", "factor75_prepared_meals",
    "soylent_subscription_receipts", "nespresso_coffee_capsule_orders", "blue_bottle_coffee_beans", "peets_coffee_club_delivery",
    "equinox_plus_fitness_app", "strava_running_subscription", "peloton_all_access_membership", "whoop_strap_membership_renewal",
    "oura_ring_sleep_insights", "apple_fitness_plus_monthly", "fitbit_premium_health_reports", "calm_app_meditation_subscription",
    "headspace_mindfulness_bill", "duolingo_max_ai_tutor", "babbel_language_learning_receipts", "grammarly_premium_subscription",
    "canva_pro_annual_invoice", "figma_organization_billing", "loom_business_video_receipts", "miro_whiteboard_workspace_bill",
    "airtable_team_plan_invoices"
]

# Ensure uniqueness
MASTER_AGENT_POOL = list(dict.fromkeys(MASTER_AGENT_POOL))

async def probe_baseline(roster: list[str], query: str) -> dict:
    """Send a query to Baseline OpenPoke using a 2-step interaction loop."""
    system_prompt = build_system_prompt()
    agents_xml = "\n".join([f'<agent name="{a}" />' for a in roster])
    content = (
        f"<conversation_history>\nNone\n</conversation_history>\n\n"
        f"<active_agents>\n{agents_xml}\n</active_agents>\n\n"
        f"<new_user_message>\n{query}\n</new_user_message>"
    )
    messages = [{"role": "user", "content": content}]

    t0 = time.perf_counter()
    selected_agent = None
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Iteration 1
    resp1 = await request_chat_completion(
        api_key=None,
        model="openai/gpt-5.6-luna",
        system=system_prompt,
        messages=messages,
        tools=TOOL_SCHEMAS,
    )
    u1 = resp1.get("usage", {})
    total_prompt_tokens += u1.get("prompt_tokens", 0)
    total_completion_tokens += u1.get("completion_tokens", 0)

    choice1 = resp1["choices"][0]["message"]
    tool_calls1 = choice1.get("tool_calls", [])

    for tc in tool_calls1:
        if tc["function"]["name"] == "send_message_to_agent":
            args = json.loads(tc["function"]["arguments"])
            selected_agent = args.get("agent_name")
            break

    # If send_message_to_agent was not called yet, provide tool response and proceed to Iteration 2
    if not selected_agent and tool_calls1:
        messages.append({
            "role": "assistant",
            "content": choice1.get("content") or "",
            "tool_calls": tool_calls1,
        })
        for tc in tool_calls1:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or tc.get("function", {}).get("name", "tool_call"),
                "content": json.dumps({"status": "delivered"}),
            })

        resp2 = await request_chat_completion(
            api_key=None,
            model="openai/gpt-5.6-luna",
            system=system_prompt,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        u2 = resp2.get("usage", {})
        total_prompt_tokens += u2.get("prompt_tokens", 0)
        total_completion_tokens += u2.get("completion_tokens", 0)

        choice2 = resp2["choices"][0]["message"]
        tool_calls2 = choice2.get("tool_calls", [])
        for tc in tool_calls2:
            if tc["function"]["name"] == "send_message_to_agent":
                args = json.loads(tc["function"]["arguments"])
                selected_agent = args.get("agent_name")
                break

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "selected_agent": selected_agent,
        "latency_ms": latency_ms,
        "input_tokens": total_prompt_tokens,
        "output_tokens": total_completion_tokens,
    }

def build_test_suite(N: int) -> tuple[list[str], list[dict]]:
    """Build a roster of exactly N agents and a balanced set of 5 test probes."""
    roster = MASTER_AGENT_POOL[:N]

    head_target = roster[min(2, len(roster)-1)]
    mid_idx = len(roster) // 2
    mid_target = roster[mid_idx]
    tail_target = roster[max(0, len(roster) - 2)]

    queries_by_name = {
        "caltrain_transit_tracker": "Check my emails for Caltrain clipper card reload receipt and ticket pass.",
        "uber_ride_receipts": "Search my emails for my recent Uber ride receipt, trip fare, and driver tip.",
        "uber_eats_receipts": "Find my latest dinner takeout receipt from Uber Eats.",
        "aws_cloud_billing": "Find my monthly cloud server compute billing invoice and EC2 charges.",
        "gcp_cloud_billing": "Check my emails for Google Cloud Platform monthly billing invoice and budget alert.",
        "azure_cloud_billing": "Search emails for Microsoft Azure subscription statements and billing alerts.",
        "github_pull_requests": "Did anyone review or comment on my GitHub pull request or tag me in an issue today?",
        "linear_issue_tracker": "Search my emails for Linear issue assignments and sprint triage updates.",
        "datadog_incident_monitor": "Search my inbox for Datadog CPU monitor alert warnings and error spikes.",
        "united_flight_concierge": "Find my United Airlines flight confirmation number and boarding pass for tomorrow.",
        "delta_flight_tracker": "Search emails for Delta SkyMiles reservation confirmations and e-tickets.",
        "marriott_hotel_receipts": "Search emails for Marriott Bonvoy hotel stay confirmations and folio receipts.",
        "amazon_delivery_tracker": "Check my emails for Amazon package shipment confirmations and tracking date.",
        "team_sync_scheduler": "Find weekly team sync calendar invite and Google Meet link for next week.",
        "netflix_subscription_manager": "Check my emails for my monthly Netflix streaming subscription receipt or plan updates.",
        "pge_electric_utility_bills": "Find my monthly electric and natural gas utility billing statement from PG&E.",
        "chase_bank_statements": "Find my Chase credit card monthly electronic statement and minimum payment due.",
        "dental_cleaning_scheduler": "Search my emails for my upcoming dental cleaning reminder and dentist confirmation.",
        "gusto_payroll_stubs": "Find my latest Gusto employee direct deposit paycheck stub and salary payment.",
        "zillow_home_listings": "Check my inbox for Zillow real estate saved search price drop notifications.",
        "ring_doorbell_motion_alerts": "Search my emails for Ring doorbell camera motion detection alert notifications.",
        "coursera_course_certificates": "Find my Coursera online course certificate of completion and receipt.",
        "steam_game_purchase_receipts": "Find my Steam game purchase receipt and digital order confirmation.",
        "homedepot_tool_rental_receipts": "Search my emails for Home Depot hardware purchase or tool rental receipt.",
    }

    probes = []

    # 1. Head probe
    h_q = queries_by_name.get(head_target, f"Find my emails related to {head_target.replace('_', ' ')}")
    probes.append({
        "type": "HEAD (Index 2)",
        "target": head_target,
        "query": h_q,
    })

    # 2. Middle probe ("Lost in the Middle")
    m_q = queries_by_name.get(mid_target, f"Search my emails for {mid_target.replace('_', ' ')}")
    probes.append({
        "type": f"MIDDLE (Index {mid_idx})",
        "target": mid_target,
        "query": m_q,
    })

    # 3. Tail probe
    t_q = queries_by_name.get(tail_target, f"Check my emails for {tail_target.replace('_', ' ')}")
    probes.append({
        "type": f"TAIL (Index {len(roster)-2})",
        "target": tail_target,
        "query": t_q,
    })

    # 4. Paraphrase probe
    if "aws_cloud_billing" in roster:
        probes.append({
            "type": "PARAPHRASE",
            "target": "aws_cloud_billing",
            "query": "Check my inbox for what I was billed for my cloud compute infrastructure and virtual machines last month.",
        })
    elif "pge_electric_utility_bills" in roster:
        probes.append({
            "type": "PARAPHRASE",
            "target": "pge_electric_utility_bills",
            "query": "How much was my home power and natural gas bill this month?",
        })

    # 5. Distractor probe (Uber Eats vs Uber Rides)
    if "uber_eats_receipts" in roster and "uber_ride_receipts" in roster:
        probes.append({
            "type": "DISTRACTOR (Uber Eats vs Ride)",
            "target": "uber_eats_receipts",
            "query": "Find how much I spent on dinner takeout from Uber last night.",
        })

    return roster, probes

async def run_sweep():
    # Sweep roster sizes N from 10 to 150
    SIZES = [10, 25, 50, 75, 100, 125, 150]

    print("==========================================================================")
    print("🔬 BASELINE ROSTER SIZE DEGRADATION SWEEP (Testing N = 10 to N = 150)")
    print("==========================================================================")

    sweep_results = []

    for N in SIZES:
        roster, probes = build_test_suite(N)
        print(f"\n--- Testing Roster Size N = {N} ({len(probes)} probes) ---")

        n_passed = 0
        n_total = len(probes)
        probe_details = []

        for p in probes:
            target = p["target"]
            query = p["query"]
            ptype = p["type"]

            res = await probe_baseline(roster, query)
            selected = res["selected_agent"]

            matched = (selected == target)
            if matched:
                n_passed += 1
                status = "✅ PASS"
            else:
                status = f"❌ FAIL (Selected: '{selected}')"

            print(f"  [{ptype:^30}] Target: '{target:^30}' -> {status} ({res['latency_ms']}ms, {res['input_tokens']} tokens)")
            probe_details.append({
                "probe_type": ptype,
                "target": target,
                "selected": selected,
                "matched": matched,
                "latency_ms": res["latency_ms"],
                "input_tokens": res["input_tokens"],
            })
            await asyncio.sleep(0.3)

        accuracy = round((n_passed / n_total) * 100, 1)
        avg_latency = round(sum(p["latency_ms"] for p in probe_details) / n_total, 1)
        avg_tokens = round(sum(p["input_tokens"] for p in probe_details) / n_total)

        print(f"  >> Score for N={N}: {n_passed}/{n_total} ({accuracy}%) | Avg Tokens: {avg_tokens} | Avg Latency: {avg_latency}ms")
        sweep_results.append({
            "roster_size": N,
            "accuracy": accuracy,
            "passed": n_passed,
            "total": n_total,
            "avg_tokens": avg_tokens,
            "avg_latency_ms": avg_latency,
            "probes": probe_details,
        })

    out_file = Path(os.getenv("DEGRADATION_REPORT_PATH", repo_dir / "docs" / "assets" / "baseline_degradation_report.json"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(sweep_results, indent=2), encoding="utf-8")
    print("\n==========================================================================")
    print(f"📊 Sweep Complete! Full report saved to {out_file}")
    print("==========================================================================")

if __name__ == "__main__":
    asyncio.run(run_sweep())
