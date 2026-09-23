#!/usr/bin/env python3
"""Seed 54 realistic execution agents across Baseline, Deterministic, and Jev rosters."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

AGENTS_SPEC = [
    # 1. Social & Creator Alerts
    ("instagram_story_monitor", "Search emails for Instagram story notifications, creator updates, and DM alerts.", ["instagram", "story", "creator", "dm", "social"]),
    ("twitter_mention_tracker", "Track Twitter/X mentions, quote tweets, and follower notification emails.", ["twitter", "tweet", "mention", "x.com", "follower"]),
    ("linkedin_networking_agent", "Monitor LinkedIn connection requests, InMail messages, and professional invitations.", ["linkedin", "inmail", "connection", "networking", "job"]),
    ("youtube_creator_notifier", "Track YouTube channel subscriber alerts, video comments, and creator studio updates.", ["youtube", "video", "subscriber", "creator", "comment"]),
    ("tiktok_creator_alert", "Scan emails for TikTok follower notifications, viral video alerts, and live stream updates.", ["tiktok", "viral", "follower", "creator", "video"]),
    ("reddit_digest_scanner", "Find Reddit reply notifications, upvote milestones, and subreddit digest emails.", ["reddit", "upvote", "subreddit", "reply", "digest"]),

    # 2. Rideshare & Transit Receipts
    ("uber_ride_receipts", "Extract Uber ride receipts, trip fares, driver tips, and route summaries from emails.", ["uber", "ride", "trip", "fare", "transit"]),
    ("lyft_transit_receipts", "Search emails for Lyft ride receipts, airport pickup charges, and transit invoices.", ["lyft", "ride", "pickup", "fare", "transit"]),
    ("caltrain_transit_tracker", "Track Caltrain clipper card reloads, monthly passes, and Bay Area transit receipts.", ["caltrain", "clipper", "transit", "train", "ticket"]),
    ("airline_baggage_fees", "Search emails for airline baggage fee receipts, seat upgrade charges, and inflight WiFi.", ["airline", "baggage", "fee", "seat", "wifi"]),
    ("parking_meter_receipts", "Find ParkMobile, PayByPhone, and airport parking lot receipt confirmations.", ["parking", "parkmobile", "meter", "garage", "receipt"]),

    # 3. Invoicing, Payments & Cloud Billing
    ("stripe_payout_notifier", "Monitor Stripe merchant payment success, customer dispute alerts, and daily payout notices.", ["stripe", "payout", "merchant", "invoice", "payment"]),
    ("paypal_payment_receipts", "Search emails for PayPal transaction receipts, seller fee invoices, and money transfer notices.", ["paypal", "payment", "transaction", "transfer", "receipt"]),
    ("aws_cloud_billing", "Find monthly Amazon Web Services (AWS) cloud billing invoices, EC2 charges, and usage alerts.", ["aws", "cloud", "billing", "amazon", "ec2", "invoice"]),
    ("gcp_cloud_billing", "Track Google Cloud Platform monthly invoices, Cloud Run usage charges, and billing budget alerts.", ["gcp", "cloud", "google", "billing", "invoice"]),
    ("azure_cloud_billing", "Search emails for Microsoft Azure subscription statements, consumption invoices, and billing alerts.", ["azure", "microsoft", "cloud", "billing", "subscription"]),
    ("chase_bank_statements", "Monitor Chase credit card e-statements, minimum payment due notices, and deposit alerts.", ["chase", "bank", "statement", "credit", "card"]),

    # 4. Developer & Engineering Workflows
    ("github_pull_requests", "Monitor GitHub pull request review requests, CI workflow failures, and issue assignments.", ["github", "pull", "request", "issue", "repo"]),
    ("gitlab_pipeline_monitor", "Track GitLab pipeline build failures, merge request approvals, and deployment notifications.", ["gitlab", "pipeline", "ci", "merge", "build"]),
    ("linear_issue_tracker", "Search emails for Linear issue assignments, sprint triage updates, and bug ticket status changes.", ["linear", "issue", "ticket", "sprint", "bug"]),
    ("jira_sprint_notifier", "Find Jira sprint planning invitations, ticket assignment notifications, and epic progress updates.", ["jira", "sprint", "epic", "ticket", "backlog"]),
    ("sentry_error_alerts", "Monitor Sentry production exception alerts, crash reports, and error spike notifications.", ["sentry", "error", "crash", "exception", "alert"]),
    ("datadog_incident_monitor", "Track Datadog infrastructure alert triggers, CPU monitor warnings, and APM error rate spikes.", ["datadog", "metric", "alert", "monitor", "cpu"]),

    # 5. Airlines, Hotels & Travel
    ("united_flight_concierge", "Track United Airlines flight booking confirmations, boarding passes, and flight delay alerts.", ["united", "flight", "airline", "boarding", "ticket"]),
    ("delta_flight_tracker", "Search emails for Delta SkyMiles reservation confirmations, e-tickets, and gate change notices.", ["delta", "flight", "skymiles", "boarding", "airline"]),
    ("airbnb_reservation_assistant", "Find Airbnb stay booking confirmations, host check-in instructions, and checkout guides.", ["airbnb", "stay", "reservation", "host", "checkin"]),
    ("marriott_hotel_receipts", "Search emails for Marriott Bonvoy hotel stay confirmations, folio receipts, and room upgrades.", ["marriott", "hotel", "bonvoy", "stay", "room"]),
    ("rental_car_receipts", "Track Hertz, Enterprise, and Avis rental car reservation confirmations and drop-off receipts.", ["rental", "car", "hertz", "enterprise", "vehicle"]),

    # 6. Food Delivery & Dining
    ("uber_eats_receipts", "Search emails for Uber Eats food delivery receipts, restaurant itemized charges, and courier tips.", ["ubereats", "food", "delivery", "restaurant", "order"]),
    ("doordash_order_tracker", "Track DoorDash order confirmations, delivery ETA updates, and merchant receipts.", ["doordash", "food", "order", "delivery", "restaurant"]),
    ("instacart_grocery_receipts", "Find Instacart grocery order receipts, out-of-stock replacements, and delivery confirmations.", ["instacart", "grocery", "order", "receipt", "supermarket"]),
    ("opentable_dining_reservations", "Track OpenTable and Resy restaurant dinner reservations, party size, and table times.", ["opentable", "resy", "reservation", "dining", "table"]),

    # 7. E-Commerce & Package Delivery
    ("amazon_delivery_tracker", "Search emails for Amazon package shipment confirmations, delivery tracking dates, and returns.", ["amazon", "package", "delivery", "shipment", "order"]),
    ("ups_package_tracker", "Track UPS shipment notifications, tracking numbers, and scheduled delivery window alerts.", ["ups", "tracking", "package", "delivery", "shipment"]),
    ("fedex_delivery_monitor", "Monitor FedEx delivery status updates, signature requirements, and tracking alerts.", ["fedex", "tracking", "package", "delivery", "signature"]),
    ("apple_store_receipts", "Find Apple Store hardware order receipts, order tracking numbers, and AppleCare invoices.", ["apple", "macbook", "iphone", "store", "hardware"]),
    ("bestbuy_order_pickup", "Search emails for Best Buy order pickup readiness confirmations and in-store pickup barcodes.", ["bestbuy", "pickup", "electronics", "store", "receipt"]),

    # 8. Calendar, Scheduling & Appointments
    ("team_sync_scheduler", "Find weekly team sync calendar invites, Google Meet video links, and recurring meeting updates.", ["team", "sync", "meeting", "calendar", "invite"]),
    ("manager_1on1_scheduler", "Monitor manager 1-on-1 meeting invites, agenda updates, and reschedule notifications.", ["1on1", "manager", "meeting", "calendar", "agenda"]),
    ("interview_scheduling_assistant", "Track candidate interview schedule confirmations, interviewer Zoom links, and debrief invites.", ["interview", "candidate", "zoom", "schedule", "debrief"]),
    ("medical_doctor_appointments", "Find doctor appointment confirmations, annual physical reminders, and clinic visit instructions.", ["doctor", "medical", "appointment", "clinic", "health"]),
    ("dental_cleaning_scheduler", "Track dental cleaning appointment reminders, dentist confirmations, and teeth cleaning schedules.", ["dental", "dentist", "cleaning", "teeth", "appointment"]),

    # 9. Subscriptions & Streaming
    ("netflix_subscription_manager", "Search emails for Netflix monthly subscription receipts, plan price changes, and new release alerts.", ["netflix", "subscription", "streaming", "receipt", "monthly"]),
    ("spotify_premium_receipts", "Track Spotify Premium monthly subscription invoices, student discounts, and playlist notifications.", ["spotify", "music", "premium", "subscription", "receipt"]),
    ("youtube_premium_receipts", "Monitor YouTube Premium monthly billing confirmations and family plan renewal notices.", ["youtube", "premium", "subscription", "billing", "renewal"]),
    ("nytimes_subscription_receipts", "Search emails for New York Times digital news subscription invoices and renewal statements.", ["nytimes", "news", "subscription", "journalism", "invoice"]),
    ("gym_fitness_membership", "Track Equinox and gym membership monthly billing statements and class booking confirmations.", ["gym", "fitness", "equinox", "membership", "workout"]),
    ("pharmacy_prescription_refills", "Monitor CVS and Walgreens prescription refill ready alerts and pharmacy order receipts.", ["cvs", "walgreens", "pharmacy", "prescription", "refill"]),

    # 10. Newsletters, Utilities & Real Estate
    ("substack_newsletter_curator", "Summarize Substack newsletter issues, tech digests, and author post notifications.", ["substack", "newsletter", "tech", "digest", "article"]),
    ("morning_brew_digest", "Find Morning Brew and business daily newsletter digest emails.", ["morningbrew", "newsletter", "business", "digest", "finance"]),
    ("gusto_payroll_stubs", "Search emails for Gusto employee payroll direct deposit stubs and tax withholding summaries.", ["gusto", "payroll", "salary", "paycheck", "deposit"]),
    ("zillow_home_listings", "Track Zillow real estate saved search alerts, open house schedules, and price drop emails.", ["zillow", "realestate", "home", "house", "listing"]),
    ("pge_electric_utility_bills", "Find PG&E monthly electric and natural gas utility billing statements and due dates.", ["pge", "electric", "gas", "utility", "bill"]),
    ("verizon_wireless_bills", "Search emails for Verizon mobile phone plan monthly bills, data usage alerts, and payment stubs.", ["verizon", "wireless", "phone", "cellular", "bill"]),
]

def execution_log_slug(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "agent"

def main() -> None:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # Paths
    base_dir = Path("/Users/akshayrakheja/Documents/general-magic-take-home")
    lab_agents_dir = base_dir / "agent-overload-evaluation-lab" / "server" / "data" / "execution_agents"
    baseline_agents_dir = base_dir / "openpoke-evaluation-baseline" / "server" / "data" / "execution_agents"

    lab_agents_dir.mkdir(parents=True, exist_ok=True)
    baseline_agents_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build AgentRecords for Enhanced (Deterministic & Jev)
    det_records = []
    jev_records = []
    baseline_names = []

    for name, purpose, tokens in AGENTS_SPEC:
        agent_id = str(uuid4())
        record_det = {
            "agent_id": agent_id,
            "name": name,
            "purpose": purpose,
            "aliases": [name, name.replace("_", " ")],
            "status": "hot",
            "use_count": 2,
            "created_at": now_iso,
            "last_used_at": now_iso,
            "memory_summary": f"Handles user queries regarding {purpose}",
            "schema_version": 1,
            "legacy_storage_key": name,
        }
        # For Jev: separate agent IDs so isolated
        record_jev = dict(record_det)
        record_jev["agent_id"] = str(uuid4())

        det_records.append(record_det)
        jev_records.append(record_jev)
        baseline_names.append(name)

        # Create baseline agent log file so use_count and history exist
        slug = execution_log_slug(name)
        log_file = baseline_agents_dir / f"{slug}.log"
        log_content = (
            f"<agent_request>{purpose}</agent_request>\n"
            f"<agent_response>Completed search for {name}. Found 3 matching records.</agent_response>\n"
        )
        log_file.write_text(log_content, encoding="utf-8")

        # Create enhanced agent log files as well
        lab_log_file = lab_agents_dir / f"{slug}.log"
        lab_log_file.write_text(log_content, encoding="utf-8")

    # 2. Write deterministic roster.json
    det_payload = {
        "schema_version": 1,
        "agents": det_records,
    }
    (lab_agents_dir / "roster.json").write_text(json.dumps(det_payload, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(det_records)} agents to Enhanced Deterministic: {lab_agents_dir / 'roster.json'}")

    # 3. Write Jev roster_jev.json
    jev_payload = {
        "schema_version": 1,
        "agents": jev_records,
    }
    (lab_agents_dir / "roster_jev.json").write_text(json.dumps(jev_payload, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(jev_records)} agents to Enhanced TypeSafe Jev: {lab_agents_dir / 'roster_jev.json'}")

    # 4. Write Baseline roster.json
    (baseline_agents_dir / "roster.json").write_text(json.dumps(baseline_names, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(baseline_names)} agents to Baseline: {baseline_agents_dir / 'roster.json'}")

if __name__ == "__main__":
    main()
