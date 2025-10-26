"""
Utility to generate a curated JSONL of manual evaluation scenarios that cover
authentication, lending, payments, KYC, disputes, reporting, and integration
edge cases. The output feeds `reports/run_manual_eval.py`.
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


BLUEPRINTS: List[Dict] = [
    {
        "prefix": "AUTH",
        "count": 10,
        "components": [
            "Authentication_Service",
            "Customer_Portal_UI",
            "Notification_Engine",
            "SMS_Service",
        ],
        "stakeholders": [
            "a mobile banking customer",
            "a treasury admin",
            "a field collections officer",
            "a fintech reseller",
            "a regional operations lead",
        ],
        "needs": [
            "need biometric fallback when reset links expire",
            "need to unlock my profile after unusual travel logins",
            "need to re-verify my identity within sixty seconds",
            "need instant OTP escalation if the authenticator app fails",
            "need adaptive MFA when device posture looks risky",
        ],
        "constraints": [
            "while commuting with spotty LTE coverage",
            "while switching between corporate VPNs",
            "while assisting customers on low-cost Android devices",
            "while onboarding merchants from shared kiosks",
            "while supporting offshore teams overnight",
        ],
        "impacts": [
            "I avoid locking out revenue-critical users",
            "support queues do not spike after fraud sweeps",
            "auditors can trace every identity decision",
            "regional launches are not blocked by SMS delays",
            "mobile CSAT scores hold steady",
        ],
        "rationale": "Stress login resilience plus outbound confirmations.",
    },
    {
        "prefix": "LOAN",
        "count": 10,
        "components": [
            "Loan_Module",
            "Credit_Score_Engine",
            "Rule_Engine",
            "Notification_Engine",
        ],
        "stakeholders": [
            "a branch lending manager",
            "a virtual loan advisor",
            "a partner bank underwriter",
            "a small-business risk analyst",
            "a servicing specialist",
        ],
        "needs": [
            "need refreshed bureau pulls before presenting offers",
            "need instant affordability checks for upsell bundles",
            "need to auto-hold risky restructures for review",
            "need to mix alternative data into scorecards",
            "need to notify brokers when counter-offers are ready",
        ],
        "constraints": [
            "when market rates jump mid-day",
            "during weekend bulk campaigns",
            "while customers navigate self-service flows",
            "during seasonal farming peaks",
            "when API quotas tighten unexpectedly",
        ],
        "impacts": [
            "we keep approval SLAs under five minutes",
            "regulators see consistent decision logic",
            "brokers stop calling for manual status",
            "we avoid issuing loans outside policy",
            "offer uptake improves for mid-market clients",
        ],
        "rationale": "Loan offers mixing scores, rules, and proactive alerts.",
    },
    {
        "prefix": "COLL",
        "count": 10,
        "components": [
            "Collections_Workflow",
            "Rule_Engine",
            "Notification_Engine",
            "Email_Communication_Module",
        ],
        "stakeholders": [
            "a recoveries director",
            "a legal collections partner",
            "an early-stage collections agent",
            "a post-chargeoff vendor",
            "a customer success lead",
        ],
        "needs": [
            "need to escalate high-risk accounts to legal automatically",
            "need softer tone templates for community lending programs",
            "need to pause outreach when disputes open",
            "need to trigger hardship plans after two returned payments",
            "need to reassign queues when agents are offline",
        ],
        "constraints": [
            "during holiday moratoriums",
            "while operating across four time zones",
            "when regional compliance wording changes midweek",
            "while managing bilingual borrower bases",
            "when SMS delivery is throttled by carriers",
        ],
        "impacts": [
            "regulators see fair-treatment evidence",
            "charge-off rates stabilize for micro-loans",
            "agents focus on accounts that matter most",
            "legal partners avoid duplicate outreach",
            "customers feel guided instead of harassed",
        ],
        "rationale": "Collections triage with policy-aware outreach.",
    },
    {
        "prefix": "KYC",
        "count": 10,
        "components": [
            "KYC_Service",
            "Document_Upload_Service",
            "Customer_Profile_Service",
            "Agent_Dashboard",
        ],
        "stakeholders": [
            "a branch KYC supervisor",
            "a refugee onboarding NGO partner",
            "a wealth onboarding concierge",
            "a cross-border payments analyst",
            "a franchise compliance officer",
        ],
        "needs": [
            "need latency-tolerant document capture",
            "need dashboards showing manual review queues",
            "need reminders when passports near expiry",
            "need to reopen applications when video calls drop",
            "need to tag politically exposed persons quickly",
        ],
        "constraints": [
            "while operating from rural cellular networks",
            "when applicants switch between web and mobile mid-flow",
            "while coordinating bilingual review teams",
            "during quarterly regulator audits",
            "when kiosk cameras downgrade image quality",
        ],
        "impacts": [
            "our SLA to activate SMEs stays under four hours",
            "regulators trust the audit trail",
            "agents know which customers need callbacks",
            "drop-off rates shrink for remote applicants",
            "credit committees spot risky profiles early",
        ],
        "rationale": "Onboarding workflows with human-in-loop visibility.",
    },
    {
        "prefix": "PAY",
        "count": 10,
        "components": [
            "Payment_Gateway",
            "API_Gateway",
            "Rule_Engine",
            "Transaction_History_Service",
        ],
        "stakeholders": [
            "a marketplace CFO",
            "an e-commerce architect",
            "a gaming studio ops lead",
            "a mobility product manager",
            "a treasury settlement analyst",
        ],
        "needs": [
            "need fewer false declines on 3DS challenges",
            "need auto-retries when issuer responses timeout",
            "need to whitelist good BIN ranges quickly",
            "need latency dashboards per acquirer",
            "need to push high-value receipts to history within seconds",
        ],
        "constraints": [
            "during holiday traffic spikes",
            "while operating multi-tenant storefronts",
            "when acquirer SLAs degrade overnight",
            "amid rolling maintenance windows",
            "while PSD2 exemptions refresh weekly",
        ],
        "impacts": [
            "checkouts stay under two seconds",
            "VIP customers stop abandoning carts",
            "finance can reconcile intraday settlements",
            "support has evidence when refunds misfire",
            "fraud teams keep clean audit logs",
        ],
        "rationale": "Payments resiliency with gateway + history signals.",
    },
    {
        "prefix": "RPT",
        "count": 10,
        "components": [
            "Reporting_Service",
            "Data_Warehouse",
            "Transaction_History_Service",
            "Customer_Profile_Service",
        ],
        "stakeholders": [
            "a finance controller",
            "a regulatory reporting lead",
            "an ecosystem partnership director",
            "a product growth analyst",
            "a sustainability program manager",
        ],
        "needs": [
            "need daily consolidated exposure reports",
            "need drilldowns by segment and delinquency",
            "need near-real-time ARR snapshots",
            "need board-ready visuals for risk committees",
            "need customer cohorts blended with payment behavior",
        ],
        "constraints": [
            "while merging multiple ERPs",
            "during quarter-end close",
            "while auditors request evidentiary samples",
            "when data source contracts are rate-limited",
            "while experimenting with new pricing levers",
        ],
        "impacts": [
            "leadership trusts the single source of truth",
            "audit findings drop year over year",
            "partners see their impact by corridor",
            "experiment teams ship faster",
            "carbon reporting aligns with finance data",
        ],
        "rationale": "Story telling dashboards across finance datasets.",
    },
    {
        "prefix": "DSP",
        "count": 10,
        "components": [
            "Dispute_Management",
            "Collections_Workflow",
            "Notification_Engine",
            "Agent_Dashboard",
        ],
        "stakeholders": [
            "a dispute analyst",
            "a merchant success manager",
            "an ombudsman liaison",
            "a chargeback operations lead",
            "a field collections coach",
        ],
        "needs": [
            "need to auto-notify collections when customers attach chargeback letters",
            "need to suppress payment reminders for accounts in dispute",
            "need SLA timers visible to every agent",
            "need to auto-escalate repeated merchant disputes",
            "need to link dispute IDs to repayment plans",
        ],
        "constraints": [
            "during schemes' shortened response windows",
            "when merchants upload bulk evidence late at night",
            "while handling multilingual attachments",
            "during regulator shadow audits",
            "when dispute queues double after outages",
        ],
        "impacts": [
            "collections does not double-contact stressed customers",
            "merchant churn drops despite spikes",
            "ombudsman cases close before breaching SLAs",
            "agents see the full context without swivel-chairing",
            "finance provisions the right reserves",
        ],
        "rationale": "Cross-team dispute visibility with alerts.",
    },
    {
        "prefix": "CORE",
        "count": 10,
        "components": [
            "Core_Banking_Integration",
            "API_Gateway",
            "Transaction_History_Service",
            "Data_Warehouse",
        ],
        "stakeholders": [
            "an integration engineer",
            "a treasury operations manager",
            "a payment rails architect",
            "a fintech BD lead",
            "a managed services vendor",
        ],
        "needs": [
            "need retries when downstream cores time out",
            "need schema drift alerts before deployments",
            "need sandbox parity with production connectors",
            "need to replay missed transactions automatically",
            "need lineage on data flowing into reports",
        ],
        "constraints": [
            "while migrating legacy SOAP services",
            "during nightly settlement batches",
            "when third parties rotate certificates unexpectedly",
            "while switching to ISO 20022 messages",
            "during hybrid-cloud cutovers",
        ],
        "impacts": [
            "transfers do not fail silently",
            "partners trust the uptime targets",
            "regulators see operational resilience plans",
            "data science teams stop rebuilding pipelines",
            "audit trails stay intact across vendors",
        ],
        "rationale": "Core integrations plus observability.",
    },
    {
        "prefix": "COMM",
        "count": 10,
        "components": [
            "Notification_Engine",
            "Email_Communication_Module",
            "SMS_Service",
            "Customer_Portal_UI",
        ],
        "stakeholders": [
            "a customer experience director",
            "a marketing automation lead",
            "a partner enablement coach",
            "a fraud communications specialist",
            "a newsroom liaison",
        ],
        "needs": [
            "need to coordinate multilingual feature rollouts",
            "need proactive outage banners inside the portal",
            "need WhatsApp opt-ins mirrored to SMS for LATAM",
            "need triggered education tips after risky events",
            "need template approvals tracked by compliance",
        ],
        "constraints": [
            "while juggling strict brand guidelines",
            "during simultaneous product launches",
            "when legal copy changes hourly",
            "while half the audience is offline overnight",
            "during crisis simulations",
        ],
        "impacts": [
            "customers trust every alert channel",
            "partners stop asking for manual updates",
            "fraud desks can prove outreach history",
            "content teams focus on strategy not ops",
            "execs see readiness before PR hits",
        ],
        "rationale": "Omni-channel comms with governance.",
    },
    {
        "prefix": "REG",
        "count": 10,
        "components": [
            "Rule_Engine",
            "Reporting_Service",
            "KYC_Service",
            "Data_Warehouse",
        ],
        "stakeholders": [
            "a chief compliance officer",
            "an AML investigator",
            "a sanctions screening lead",
            "a risk technology manager",
            "a privacy program owner",
        ],
        "needs": [
            "need nightly reviews of high-risk entities",
            "need to evidence every adverse-media rule hit",
            "need to flag onboarding trends by jurisdiction",
            "need to push escalations straight to regulators",
            "need to auto-close alerts with duplicate docs",
        ],
        "constraints": [
            "during regulator onsite exams",
            "while running model validations",
            "when whistleblower hotlines spike",
            "while migrating to new policy thresholds",
            "during privacy data-retention purges",
        ],
        "impacts": [
            "board packs cite the same metrics every month",
            "investigators stop drowning in false positives",
            "regulators see proactive posture",
            "legal holds are traceable in reports",
            "privacy teams trust suppression workflows",
        ],
        "rationale": "Regulatory control room scenarios.",
    },
]


def build_cases(limit: int | None = None) -> List[Dict]:
    cases: List[Dict] = []
    rng = random.Random(42)
    for blueprint in BLUEPRINTS:
        combos = product(
            blueprint["stakeholders"],
            blueprint["needs"],
            blueprint["constraints"],
            blueprint["impacts"],
        )
        combo_list = list(combos)
        rng.shuffle(combo_list)
        for idx, (stakeholder, need, constraint, impact) in enumerate(combo_list[: blueprint["count"]]):
            text = (
                f"As {stakeholder}, I {need} {constraint}, so that {impact}."
            )
            cases.append(
                {
                    "id": f"{blueprint['prefix']}{idx+1:02d}",
                    "text": text,
                    "expected": blueprint["components"],
                    "rationale": f"{blueprint['rationale']} Stakeholder focus: {stakeholder}.",
                }
            )
    if limit:
        cases = cases[:limit]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manual evaluation cases JSONL.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "manual_eval_cases_100.jsonl",
    )
    parser.add_argument("--limit", type=int, default=100, help="Trim the dataset to N scenarios.")
    args = parser.parse_args()

    cases = build_cases(limit=args.limit)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fp:
        for case in cases:
            fp.write(json.dumps(case) + "\n")
    print(f"Wrote {len(cases)} cases to {args.output_path}")


if __name__ == "__main__":
    main()
