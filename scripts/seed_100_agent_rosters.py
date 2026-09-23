#!/usr/bin/env python3
"""Seed 100 realistic execution agents across Baseline, Deterministic, and Jev rosters.

Also clears previous conversation logs across all 3 systems to provide a clean 15-turn test.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

AGENTS_100 = [
    # 1. Rideshare & Transit (6)
    ("uber_ride_receipts", "Extract Uber ride receipts, trip fares, driver tips, and route summaries from emails.", ["uber", "ride", "trip", "fare", "transit"]),
    ("lyft_transit_receipts", "Search emails for Lyft ride receipts, airport pickup charges, and transit invoices.", ["lyft", "ride", "pickup", "fare", "transit"]),
    ("caltrain_transit_tracker", "Track Caltrain clipper card reloads, monthly passes, and Bay Area transit receipts.", ["caltrain", "clipper", "transit", "train", "ticket"]),
    ("waymo_autonomous_trips", "Search emails for Waymo autonomous vehicle ride receipts and trip fares.", ["waymo", "autonomous", "ride", "trip", "fare"]),
    ("lime_scooter_receipts", "Track Lime and Bird electric scooter ride charges, unlocks, and wallet reloads.", ["lime", "bird", "scooter", "ride", "unlock"]),
    ("parking_meter_receipts", "Find ParkMobile, PayByPhone, and airport parking lot receipt confirmations.", ["parking", "parkmobile", "meter", "garage", "receipt"]),

    # 2. Food Delivery & Dining (6)
    ("uber_eats_receipts", "Search emails for Uber Eats food delivery receipts, restaurant itemized charges, and courier tips.", ["ubereats", "food", "delivery", "restaurant", "order"]),
    ("doordash_order_tracker", "Track DoorDash order confirmations, delivery ETA updates, and merchant receipts.", ["doordash", "food", "order", "delivery", "restaurant"]),
    ("instacart_grocery_receipts", "Find Instacart grocery order receipts, out-of-stock replacements, and delivery confirmations.", ["instacart", "grocery", "order", "receipt", "supermarket"]),
    ("opentable_dining_reservations", "Track OpenTable and Resy restaurant dinner reservations, party size, and table times.", ["opentable", "resy", "reservation", "dining", "table"]),
    ("seamless_grubhub_delivery", "Search emails for Grubhub and Seamless food delivery orders and restaurant receipts.", ["grubhub", "seamless", "food", "delivery", "restaurant"]),
    ("starbucks_mobile_orders", "Find Starbucks mobile app order receipts, card balance reloads, and reward star updates.", ["starbucks", "coffee", "mobile", "order", "stars"]),

    # 3. Cloud Infrastructure & Invoicing (8)
    ("aws_cloud_billing", "Find monthly Amazon Web Services (AWS) cloud billing invoices, EC2 charges, and usage alerts.", ["aws", "cloud", "billing", "amazon", "ec2", "invoice"]),
    ("gcp_cloud_billing", "Track Google Cloud Platform monthly invoices, Cloud Run usage charges, and billing budget alerts.", ["gcp", "cloud", "google", "billing", "invoice"]),
    ("azure_cloud_billing", "Search emails for Microsoft Azure subscription statements, consumption invoices, and billing alerts.", ["azure", "microsoft", "cloud", "billing", "subscription"]),
    ("stripe_payout_notifier", "Monitor Stripe merchant payment success, customer dispute alerts, and daily payout notices.", ["stripe", "payout", "merchant", "invoice", "payment"]),
    ("paypal_payment_receipts", "Search emails for PayPal transaction receipts, seller fee invoices, and money transfer notices.", ["paypal", "payment", "transaction", "transfer", "receipt"]),
    ("vercel_usage_invoices", "Track Vercel deployment bandwidth invoices, team member seat charges, and serverless usage.", ["vercel", "deployment", "bandwidth", "invoice", "serverless"]),
    ("openai_api_invoices", "Search emails for OpenAI API credit purchases, monthly usage tier receipts, and billing updates.", ["openai", "api", "tokens", "credits", "invoice"]),
    ("snowflake_cloud_credits", "Monitor Snowflake data cloud compute credit usage, capacity invoices, and warehouse cost alerts.", ["snowflake", "warehouse", "compute", "credits", "billing"]),

    # 4. Developer Tools & DevOps (8)
    ("github_pull_requests", "Monitor GitHub pull request review requests, CI workflow failures, and issue assignments.", ["github", "pull", "request", "issue", "repo"]),
    ("gitlab_pipeline_monitor", "Track GitLab pipeline build failures, merge request approvals, and deployment notifications.", ["gitlab", "pipeline", "ci", "merge", "build"]),
    ("linear_issue_tracker", "Search emails for Linear issue assignments, sprint triage updates, and bug ticket status changes.", ["linear", "issue", "ticket", "sprint", "bug"]),
    ("jira_sprint_notifier", "Find Jira sprint planning invitations, ticket assignment notifications, and epic progress updates.", ["jira", "sprint", "epic", "ticket", "backlog"]),
    ("sentry_error_alerts", "Monitor Sentry production exception alerts, crash reports, and error spike notifications.", ["sentry", "error", "crash", "exception", "alert"]),
    ("datadog_incident_monitor", "Track Datadog infrastructure alert triggers, CPU monitor warnings, and APM error rate spikes.", ["datadog", "metric", "alert", "monitor", "cpu"]),
    ("pagerduty_incident_alerts", "Search emails for PagerDuty on-call alert escalations, incident resolutions, and shift schedules.", ["pagerduty", "oncall", "incident", "escalation", "alert"]),
    ("docker_hub_build_status", "Track Docker Hub automated container build notices, security vulnerability scans, and tag pushes.", ["docker", "container", "build", "image", "hub"]),

    # 5. Airlines & Travel (8)
    ("united_flight_concierge", "Track United Airlines flight booking confirmations, boarding passes, and flight delay alerts.", ["united", "flight", "airline", "boarding", "ticket"]),
    ("delta_flight_tracker", "Search emails for Delta SkyMiles reservation confirmations, e-tickets, and gate change notices.", ["delta", "flight", "skymiles", "boarding", "airline"]),
    ("alaska_airlines_travel", "Find Alaska Airlines flight itineraries, mileage plan updates, and boarding pass emails.", ["alaska", "flight", "airline", "itinerary", "mileage"]),
    ("american_airlines_flights", "Monitor American Airlines AAdvantage flight bookings, seat assignment updates, and gate changes.", ["american", "aa", "flight", "aadvantage", "gate"]),
    ("airbnb_reservation_assistant", "Find Airbnb stay booking confirmations, host check-in instructions, and checkout guides.", ["airbnb", "stay", "reservation", "host", "checkin"]),
    ("marriott_hotel_receipts", "Search emails for Marriott Bonvoy hotel stay confirmations, folio receipts, and room upgrades.", ["marriott", "hotel", "bonvoy", "stay", "room"]),
    ("hilton_honors_reservations", "Track Hilton Honors hotel reservations, digital key arrival notices, and stay point statements.", ["hilton", "honors", "hotel", "reservation", "points"]),
    ("rental_car_receipts", "Track Hertz, Enterprise, and Avis rental car reservation confirmations and drop-off receipts.", ["rental", "car", "hertz", "enterprise", "vehicle"]),

    # 6. E-Commerce & Logistics (8)
    ("amazon_delivery_tracker", "Search emails for Amazon package shipment confirmations, delivery tracking dates, and returns.", ["amazon", "package", "delivery", "shipment", "order"]),
    ("ups_package_tracker", "Track UPS shipment notifications, tracking numbers, and scheduled delivery window alerts.", ["ups", "tracking", "package", "delivery", "shipment"]),
    ("fedex_delivery_monitor", "Monitor FedEx delivery status updates, signature requirements, and tracking alerts.", ["fedex", "tracking", "package", "delivery", "signature"]),
    ("usps_informed_delivery", "Check daily USPS mail preview digests, package tracking numbers, and delivery confirmations.", ["usps", "mail", "delivery", "tracking", "post"]),
    ("apple_store_receipts", "Find Apple Store hardware order receipts, order tracking numbers, and AppleCare invoices.", ["apple", "macbook", "iphone", "store", "hardware"]),
    ("bestbuy_order_pickup", "Search emails for Best Buy order pickup readiness confirmations and in-store pickup barcodes.", ["bestbuy", "pickup", "electronics", "store", "receipt"]),
    ("target_circle_receipts", "Track Target drive-up pickup notifications, in-store receipt digests, and Target Circle savings.", ["target", "pickup", "driveup", "store", "receipt"]),
    ("costco_order_deliveries", "Search emails for Costco.com warehouse item orders, tracking numbers, and shipment invoices.", ["costco", "warehouse", "order", "delivery", "shipment"]),

    # 7. Calendar & Scheduling (6)
    ("team_sync_scheduler", "Find weekly team sync calendar invites, Google Meet video links, and recurring meeting updates.", ["team", "sync", "meeting", "calendar", "invite"]),
    ("manager_1on1_scheduler", "Monitor manager 1-on-1 meeting invites, agenda updates, and reschedule notifications.", ["1on1", "manager", "meeting", "calendar", "agenda"]),
    ("interview_scheduling_assistant", "Track candidate interview schedule confirmations, interviewer Zoom links, and debrief invites.", ["interview", "candidate", "zoom", "schedule", "debrief"]),
    ("zoom_meeting_recordings", "Find Zoom cloud recording availability notices, meeting passcode links, and audio transcripts.", ["zoom", "recording", "transcript", "meeting", "passcode"]),
    ("calendly_booking_notifications", "Track Calendly meeting bookings, attendee questionnaire answers, and cancellation notices.", ["calendly", "booking", "schedule", "attendee", "meeting"]),
    ("doodle_poll_organizer", "Monitor Doodle group poll responses, participant availability, and finalized meeting times.", ["doodle", "poll", "meeting", "availability", "schedule"]),

    # 8. Subscriptions & Streaming (8)
    ("netflix_subscription_manager", "Search emails for Netflix monthly subscription receipts, plan price changes, and new release alerts.", ["netflix", "subscription", "streaming", "receipt", "monthly"]),
    ("spotify_premium_receipts", "Track Spotify Premium monthly subscription invoices, student discounts, and playlist notifications.", ["spotify", "music", "premium", "subscription", "receipt"]),
    ("youtube_premium_receipts", "Monitor YouTube Premium monthly billing confirmations and family plan renewal notices.", ["youtube", "premium", "subscription", "billing", "renewal"]),
    ("nytimes_subscription_receipts", "Search emails for New York Times digital news subscription invoices and renewal statements.", ["nytimes", "news", "subscription", "journalism", "invoice"]),
    ("hulu_disney_bundle_billing", "Track Disney+ and Hulu monthly bundle subscription statements and pricing change notices.", ["disney", "hulu", "streaming", "bundle", "subscription"]),
    ("chatgpt_plus_receipts", "Search emails for OpenAI ChatGPT Plus monthly subscription charges and tax invoices.", ["chatgpt", "openai", "plus", "subscription", "receipt"]),
    ("adobe_creative_cloud", "Find Adobe Creative Cloud photography plan annual renewals, license receipts, and app updates.", ["adobe", "creative", "cloud", "photoshop", "subscription"]),
    ("apple_services_billing", "Search emails for Apple iCloud+, Apple TV+, and App Store monthly subscription receipt invoices.", ["icloud", "apple", "appstore", "subscription", "billing"]),

    # 9. Healthcare & Wellness (6)
    ("medical_doctor_appointments", "Find doctor appointment confirmations, annual physical reminders, and clinic visit instructions.", ["doctor", "medical", "appointment", "clinic", "health"]),
    ("dental_cleaning_scheduler", "Track dental cleaning appointment reminders, dentist confirmations, and teeth cleaning schedules.", ["dental", "dentist", "cleaning", "teeth", "appointment"]),
    ("pharmacy_prescription_refills", "Monitor CVS and Walgreens prescription refill ready alerts and pharmacy order receipts.", ["cvs", "walgreens", "pharmacy", "prescription", "refill"]),
    ("gym_fitness_membership", "Track Equinox and gym membership monthly billing statements and class booking confirmations.", ["gym", "fitness", "equinox", "membership", "workout"]),
    ("optometry_eye_exam_tracker", "Search emails for vision insurance claims, optometrist eye exam appointments, and contact lens orders.", ["optometry", "vision", "eye", "exam", "contacts"]),
    ("labcorp_blood_test_results", "Find Labcorp and Quest Diagnostics specimen collection confirmations and test result ready alerts.", ["labcorp", "quest", "blood", "test", "results"]),

    # 10. Utilities & Telecom (6)
    ("pge_electric_utility_bills", "Find PG&E monthly electric and natural gas utility billing statements and due dates.", ["pge", "electric", "gas", "utility", "bill"]),
    ("coned_gas_statements", "Track ConEdison residential natural gas usage statements, meter readings, and bill payment stubs.", ["coned", "gas", "utility", "meter", "statement"]),
    ("verizon_wireless_bills", "Search emails for Verizon mobile phone plan monthly bills, data usage alerts, and payment stubs.", ["verizon", "wireless", "phone", "cellular", "bill"]),
    ("tmobile_family_plan", "Monitor T-Mobile monthly family plan bills, device payment installments, and AutoPay notices.", ["tmobile", "cellular", "family", "plan", "bill"]),
    ("comcast_xfinity_monthly_bill", "Track Comcast Xfinity home internet speed tier bills, modem rental fees, and payment receipts.", ["comcast", "xfinity", "internet", "modem", "bill"]),
    ("city_water_trash_utility", "Search emails for municipal city residential water, sewer, and garbage collection utility bills.", ["water", "sewer", "trash", "utility", "city"]),

    # 11. Banking, Taxes & Investments (8)
    ("chase_bank_statements", "Monitor Chase credit card e-statements, minimum payment due notices, and deposit alerts.", ["chase", "bank", "statement", "credit", "card"]),
    ("bank_of_america_alerts", "Search emails for Bank of America checking balance alerts, wire confirmations, and fraud notices.", ["bofa", "bank", "checking", "wire", "balance"]),
    ("fidelity_401k_contributions", "Track Fidelity employer 401(k) retirement contributions, statement digests, and trade confirms.", ["fidelity", "401k", "retirement", "contribution", "investment"]),
    ("vanguard_index_fund_dividends", "Search emails for Vanguard brokerage dividend reinvestments, distribution notices, and tax forms.", ["vanguard", "dividend", "index", "brokerage", "distribution"]),
    ("robinhood_stock_trade_confirms", "Track Robinhood stock and ETF trade executions, crypto deposits, and monthly account summaries.", ["robinhood", "stock", "trade", "etf", "crypto"]),
    ("turbotax_tax_filing_status", "Search emails for TurboTax federal and state tax return acceptance notices and refund deposits.", ["turbotax", "tax", "irs", "return", "refund"]),
    ("amex_membership_rewards", "Monitor American Express monthly statements, membership reward point balances, and card fee receipts.", ["amex", "americanexpress", "rewards", "points", "statement"]),
    ("wells_fargo_mortgage_statements", "Find Wells Fargo home mortgage monthly escrow statements, interest summaries, and payment notices.", ["wellsfargo", "mortgage", "escrow", "home", "loan"]),

    # 12. Social & Creator Media (6)
    ("instagram_story_monitor", "Search emails for Instagram story notifications, creator updates, and DM alerts.", ["instagram", "story", "creator", "dm", "social"]),
    ("twitter_mention_tracker", "Track Twitter/X mentions, quote tweets, and follower notification emails.", ["twitter", "tweet", "mention", "x.com", "follower"]),
    ("linkedin_networking_agent", "Monitor LinkedIn connection requests, InMail messages, and professional invitations.", ["linkedin", "inmail", "connection", "networking", "job"]),
    ("youtube_creator_notifier", "Track YouTube channel subscriber alerts, video comments, and creator studio updates.", ["youtube", "video", "subscriber", "creator", "comment"]),
    ("tiktok_creator_alert", "Scan emails for TikTok follower notifications, viral video alerts, and live stream updates.", ["tiktok", "viral", "follower", "creator", "video"]),
    ("reddit_digest_scanner", "Find Reddit reply notifications, upvote milestones, and subreddit digest emails.", ["reddit", "upvote", "subreddit", "reply", "digest"]),

    # 13. Smart Home & Security (4)
    ("ring_doorbell_motion_alerts", "Search emails for Ring video doorbell motion alerts, package detection, and cloud subscription bills.", ["ring", "doorbell", "camera", "motion", "security"]),
    ("nest_thermostat_energy_report", "Track Google Nest thermostat monthly home energy savings reports and seasonal temperature tips.", ["nest", "thermostat", "energy", "hvac", "temperature"]),
    ("arlo_security_camera_events", "Find Arlo outdoor security camera battery low warnings and recorded motion event alerts.", ["arlo", "camera", "security", "battery", "motion"]),
    ("philips_hue_lighting_schedules", "Monitor Philips Hue smart bulb bridge connection notices and lighting automation updates.", ["hue", "philips", "lighting", "bulb", "smart"]),

    # 14. Education & Online Learning (4)
    ("coursera_course_certificates", "Find Coursera specialization certificate completions, course assignment grades, and tuition receipts.", ["coursera", "course", "certificate", "education", "grade"]),
    ("duolingo_streak_reminders", "Track Duolingo language practice streak reminders, league rankings, and Super Duolingo receipts.", ["duolingo", "streak", "language", "practice", "lesson"]),
    ("udemy_course_receipts", "Search emails for Udemy coding course purchases, instructor announcements, and lifetime access confirmations.", ["udemy", "course", "coding", "tutorial", "receipt"]),
    ("submittable_literary_submissions", "Monitor Submittable literary journal submission status changes, acceptance letters, and decline notices.", ["submittable", "journal", "submission", "acceptance", "manuscript"]),

    # 15. Work, Payroll & HR (4)
    ("gusto_payroll_stubs", "Search emails for Gusto employee payroll direct deposit stubs and tax withholding summaries.", ["gusto", "payroll", "salary", "paycheck", "deposit"]),
    ("adp_tax_w2_statements", "Find ADP annual Form W-2 electronic tax statements, end-of-year earnings summaries, and 1099 forms.", ["adp", "w2", "tax", "earnings", "payroll"]),
    ("greenhouse_applicant_reviews", "Track Greenhouse candidate interview scorecards, applicant stage transitions, and referral notifications.", ["greenhouse", "candidate", "interview", "recruiting", "applicant"]),
    ("workday_timeoff_requests", "Search emails for Workday PTO vacation requests, manager approvals, and floating holiday balances.", ["workday", "pto", "vacation", "timeoff", "approval"]),

    # 16. Real Estate & Housing (4)
    ("zillow_home_listings", "Track Zillow real estate saved search alerts, open house schedules, and price drop emails.", ["zillow", "realestate", "home", "house", "listing"]),
    ("redfin_price_drop_alerts", "Search emails for Redfin instant home tour schedule confirmations and neighborhood market reports.", ["redfin", "tour", "home", "price", "market"]),
    ("apartments_com_tour_requests", "Find Apartments.com rental unit inquiry replies, landlord touring slots, and application links.", ["apartments", "rental", "tour", "lease", "landlord"]),
    ("hoa_monthly_assessment_notices", "Track Homeowners Association (HOA) monthly dues assessment invoices and community meeting notices.", ["hoa", "assessment", "dues", "association", "condo"]),
]

def execution_log_slug(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "agent"

def main() -> None:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    repo_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(os.getenv("EXPERIMENT_ROOT_DIR", repo_dir.parent))
    lab_agents_dir = Path(os.getenv("LAB_AGENTS_DIR", repo_dir / "server" / "data" / "execution_agents"))
    baseline_agents_dir = Path(os.getenv("BASELINE_AGENTS_DIR", base_dir / "openpoke-evaluation-baseline" / "server" / "data" / "execution_agents"))
    baseline_conv_dir = Path(os.getenv("BASELINE_CONV_DIR", base_dir / "openpoke-evaluation-baseline" / "server" / "data" / "conversation"))
    lab_conv_dir = Path(os.getenv("LAB_CONV_DIR", repo_dir / "server" / "data" / "conversation"))

    lab_agents_dir.mkdir(parents=True, exist_ok=True)
    baseline_agents_dir.mkdir(parents=True, exist_ok=True)
    baseline_conv_dir.mkdir(parents=True, exist_ok=True)
    lab_conv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clear previous conversation logs across all 3 systems
    (baseline_conv_dir / "poke_conversation.log").write_text("", encoding="utf-8")
    (baseline_conv_dir / "working_memory.log").write_text("", encoding="utf-8")
    for f in lab_conv_dir.glob("*.log"):
        f.write_text("", encoding="utf-8")
    print("✓ Cleared all historical conversation logs across Baseline, Deterministic, and Jev.")

    # 2. Build exactly 100 AgentRecords
    assert len(AGENTS_100) == 100, f"Expected exactly 100 agents, got {len(AGENTS_100)}"

    det_records = []
    jev_records = []
    baseline_names = []

    for name, purpose, tokens in AGENTS_100:
        agent_id_det = str(uuid4())
        agent_id_jev = str(uuid4())

        record_det = {
            "agent_id": agent_id_det,
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

        record_jev = dict(record_det)
        record_jev["agent_id"] = agent_id_jev

        det_records.append(record_det)
        jev_records.append(record_jev)
        baseline_names.append(name)

        slug = execution_log_slug(name)
        log_content = (
            f"<agent_request>{purpose}</agent_request>\n"
            f"<agent_response>Completed task search for {name}. Found records.</agent_response>\n"
        )
        (baseline_agents_dir / f"{slug}.log").write_text(log_content, encoding="utf-8")
        (lab_agents_dir / f"{slug}.log").write_text(log_content, encoding="utf-8")

    # Write rosters
    det_payload = {"schema_version": 1, "agents": det_records}
    (lab_agents_dir / "roster.json").write_text(json.dumps(det_payload, indent=2), encoding="utf-8")

    jev_payload = {"schema_version": 1, "agents": jev_records}
    (lab_agents_dir / "roster_jev.json").write_text(json.dumps(jev_payload, indent=2), encoding="utf-8")

    (baseline_agents_dir / "roster.json").write_text(json.dumps(baseline_names, indent=2), encoding="utf-8")

    print(f"✓ Seeded 100 agents to Baseline: {baseline_agents_dir / 'roster.json'}")
    print(f"✓ Seeded 100 agents to Enhanced Deterministic: {lab_agents_dir / 'roster.json'}")
    print(f"✓ Seeded 100 agents to Enhanced TypeSafe Jev: {lab_agents_dir / 'roster_jev.json'}")

if __name__ == "__main__":
    main()
