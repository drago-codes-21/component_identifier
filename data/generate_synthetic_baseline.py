"""
Generate a balanced synthetic training dataset for component identification.

The script emits 3,000 user stories covering 25 recurring servicing scenarios.
Every scenario maps to one or more components from the official taxonomy so
DistilBERT receives consistent supervision even before real data is available.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

SCENARIOS = [
    {
        "trigger": "keep getting complaints that OTP codes never arrive during password resets",
        "need": "stabilize the authentication flow across SMS and email delivery paths",
        "components": [
            "Authentication_Service",
            "SMS_Service",
            "Email_Communication_Module",
            "Customer_Portal_UI",
        ],
    },
    {
        "trigger": "see borrowers drop out because their uploaded proofs are rejected without context",
        "need": "improve document capture, validation, and follow-up tasks",
        "components": [
            "Document_Upload_Service",
            "KYC_Service",
            "Customer_Profile_Service",
            "Notification_Engine",
        ],
    },
    {
        "trigger": "manage delinquent queues manually in spreadsheets",
        "need": "advance the collections workflow with automated routing",
        "components": [
            "Collections_Workflow",
            "Agent_Dashboard",
            "Rule_Engine",
        ],
    },
    {
        "trigger": "field teams cannot generate accurate restructure quotes for borrowers",
        "need": "provide self-serve loan adjustments tied to payments and reporting",
        "components": [
            "Loan_Module",
            "Payment_Gateway",
            "Reporting_Service",
            "Customer_Profile_Service",
        ],
    },
    {
        "trigger": "partner APIs spike traffic during campaigns and throttle our services",
        "need": "enforce policies and alert squads via the gateway",
        "components": [
            "API_Gateway",
            "Notification_Engine",
            "Rule_Engine",
        ],
    },
    {
        "trigger": "credit decisions take hours while we wait on batch jobs",
        "need": "score applicants in real time with core banking context",
        "components": [
            "Credit_Score_Engine",
            "Core_Banking_Integration",
            "Data_Warehouse",
        ],
    },
    {
        "trigger": "disputes linger because agents lack a full payment trail",
        "need": "surface transaction evidence and streamline investigation messaging",
        "components": [
            "Dispute_Management",
            "Transaction_History_Service",
            "Email_Communication_Module",
        ],
    },
    {
        "trigger": "customers cannot trace historic payments inside the portal",
        "need": "expose interactive timelines and ledger level context",
        "components": [
            "Customer_Portal_UI",
            "Transaction_History_Service",
            "Data_Warehouse",
        ],
    },
    {
        "trigger": "auto-debit failures go unnoticed until merchants escalate",
        "need": "monitor payment pipelines and push proactive alerts",
        "components": [
            "Payment_Gateway",
            "Notification_Engine",
            "SMS_Service",
        ],
    },
    {
        "trigger": "profile updates lag between branches and digital channels",
        "need": "sync golden customer records with downstream systems",
        "components": [
            "Customer_Profile_Service",
            "Data_Warehouse",
            "API_Gateway",
        ],
    },
    {
        "trigger": "KYC renewals slip through the cracks each quarter",
        "need": "schedule reminders across channels with audit visibility",
        "components": [
            "KYC_Service",
            "Notification_Engine",
            "Email_Communication_Module",
            "SMS_Service",
        ],
    },
    {
        "trigger": "agents juggle multiple tabs to track promises-to-pay",
        "need": "centralize KPIs and tasks in their dashboard",
        "components": [
            "Agent_Dashboard",
            "Collections_Workflow",
            "Reporting_Service",
        ],
    },
    {
        "trigger": "borrowers abandon document upload on mobile devices",
        "need": "optimize capture, confirmations, and nudges",
        "components": [
            "Document_Upload_Service",
            "Customer_Portal_UI",
            "Notification_Engine",
        ],
    },
    {
        "trigger": "finance teams cannot auto-generate payoff letters",
        "need": "build templates linked to loan balances and communications",
        "components": [
            "Loan_Module",
            "Reporting_Service",
            "Email_Communication_Module",
        ],
    },
    {
        "trigger": "dispute analysts struggle to collect supporting evidence",
        "need": "request and organize attachments tied to cases",
        "components": [
            "Dispute_Management",
            "Document_Upload_Service",
            "Notification_Engine",
        ],
    },
    {
        "trigger": "core banking updates arrive late to partner APIs",
        "need": "streamline integrations with monitoring and history replay",
        "components": [
            "API_Gateway",
            "Core_Banking_Integration",
            "Transaction_History_Service",
        ],
    },
    {
        "trigger": "risk teams lack cohesive scorecards on portfolio health",
        "need": "combine credit scores, warehouse aggregates, and business logic",
        "components": [
            "Reporting_Service",
            "Data_Warehouse",
            "Credit_Score_Engine",
            "Rule_Engine",
        ],
    },
    {
        "trigger": "payment receipts are inconsistent across channels",
        "need": "trigger standardized confirmations with transaction context",
        "components": [
            "Payment_Gateway",
            "Email_Communication_Module",
            "Transaction_History_Service",
        ],
    },
    {
        "trigger": "high-balance delinquencies do not escalate fast enough",
        "need": "auto-flag risky accounts and prioritize worklists",
        "components": [
            "Collections_Workflow",
            "Rule_Engine",
            "Agent_Dashboard",
        ],
    },
    {
        "trigger": "SMS dunning campaigns go out with stale balances",
        "need": "merge workflow statuses with real-time notifications",
        "components": [
            "SMS_Service",
            "Notification_Engine",
            "Collections_Workflow",
        ],
    },
    {
        "trigger": "warehouse refreshes fail to keep up with daily transaction volumes",
        "need": "stabilize data ingestion from core banking feeds",
        "components": [
            "Data_Warehouse",
            "Core_Banking_Integration",
            "Transaction_History_Service",
        ],
    },
    {
        "trigger": "customers cannot schedule installments from self-service channels",
        "need": "enable plan creation tied to loan terms and payments",
        "components": [
            "Customer_Portal_UI",
            "Loan_Module",
            "Payment_Gateway",
        ],
    },
    {
        "trigger": "support lacks alerts when critical payments fail",
        "need": "notify agents instantly with customer context",
        "components": [
            "Notification_Engine",
            "SMS_Service",
            "Agent_Dashboard",
        ],
    },
    {
        "trigger": "customers are left in the dark during dispute investigations",
        "need": "orchestrate multi-channel updates until resolution",
        "components": [
            "Dispute_Management",
            "Email_Communication_Module",
            "SMS_Service",
        ],
    },
    {
        "trigger": "suspicious login attempts go unchallenged overnight",
        "need": "enforce adaptive rules and broadcast security alerts",
        "components": [
            "Authentication_Service",
            "Rule_Engine",
            "Notification_Engine",
        ],
    },
]

ROLES = [
    "collections supervisor",
    "credit risk analyst",
    "customer success lead",
    "operations manager",
    "support agent",
    "regional collections head",
    "product manager",
    "payments specialist",
    "field officer",
    "digital servicing director",
]

URGENCIES = [
    "during end-of-month spikes",
    "while handling regulatory reviews",
    "before the next release window",
    "as we onboard new partners",
    "when outages hit after hours",
    "to hit this quarter's KPIs",
]

GOALS = [
    "so borrowers regain trust quickly",
    "to shrink manual escalations",
    "to stay compliant without firefighting",
    "so we meet SLA targets",
    "to give leadership real-time visibility",
    "so agents can focus on negotiations",
]

CHANNELS = [
    "through the mobile portal",
    "inside the agent workspace",
    "via automated campaigns",
    "through partner integrations",
    "inside scheduled reporting packs",
    "through proactive alerts",
]


def build_story(trigger: str, need: str) -> str:
    role = random.choice(ROLES)
    urgency = random.choice(URGENCIES)
    goal = random.choice(GOALS)
    channel = random.choice(CHANNELS)
    tone = random.choice([" urgently", "", " quickly"])
    return (
        f"As a {role}, I {trigger} {urgency}. "
        f"I need to {need} {channel} {goal}{tone}."
    )


def main(target_path: Path, samples_per_scenario: int = 120) -> None:
    rows = []
    for scenario in SCENARIOS:
        for _ in range(samples_per_scenario):
            text = build_story(scenario["trigger"], scenario["need"])
            labels = "|".join(scenario["components"])
            rows.append((text, labels))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "labels"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic training rows to {target_path}")


if __name__ == "__main__":
    OUTPUT = Path(__file__).resolve().parent / "train.csv"
    main(OUTPUT)
