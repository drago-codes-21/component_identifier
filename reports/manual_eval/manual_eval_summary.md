# Manual Scenario Evaluation

- Source cases: `reports\manual_eval_cases.jsonl`
- Total scenarios reviewed: 10
- Precision 50.0% (model called 8 components correctly and 8 incorrectly).
- Recall 24.2% (found 8 of 33 expected components).
- Exact matches: 0/10.
- Most frequently missed components: Notification_Engine (3), Email_Communication_Module (2), Rule_Engine (2), Customer_Portal_UI (2), KYC_Service (2).

## Scenario Breakdown
- CASE01 [PARTIAL] expected Authentication_Service, Customer_Portal_UI, Notification_Engine, SMS_Service, Email_Communication_Module; model predicted Authentication_Service, Customer_Portal_UI, SMS_Service. Hits: Authentication_Service, Customer_Portal_UI, SMS_Service. Extra: none. Missed: Email_Communication_Module, Notification_Engine.
  - Why it matters: Common self-service flow touching auth UI plus outbound comms.
- CASE02 [MISS] expected Credit_Score_Engine, Loan_Module, Rule_Engine; model predicted Data_Warehouse. Hits: none. Extra: Data_Warehouse. Missed: Credit_Score_Engine, Loan_Module, Rule_Engine.
  - Why it matters: Risk-focused feature requiring scoring + decisioning.
- CASE03 [PARTIAL] expected Notification_Engine, SMS_Service, Email_Communication_Module; model predicted Notification_Engine, SMS_Service. Hits: Notification_Engine, SMS_Service. Extra: none. Missed: Email_Communication_Module.
  - Why it matters: Stress-test multi-channel messaging plus regional context.
- CASE04 [MISS] expected Document_Upload_Service, Customer_Portal_UI, KYC_Service; model predicted Data_Warehouse. Hits: none. Extra: Data_Warehouse. Missed: Customer_Portal_UI, Document_Upload_Service, KYC_Service.
  - Why it matters: Edge case around degraded networks and KYC artifacts.
- CASE05 [MISS] expected Payment_Gateway, Rule_Engine, API_Gateway; model predicted Transaction_History_Service. Hits: none. Extra: Transaction_History_Service. Missed: API_Gateway, Payment_Gateway, Rule_Engine.
  - Why it matters: Gateway + rule tuning scenario; checks precision under similar intents.
- CASE06 [PARTIAL] expected Reporting_Service, Data_Warehouse, Collections_Workflow, Transaction_History_Service; model predicted Email_Communication_Module, Transaction_History_Service. Hits: Transaction_History_Service. Extra: Email_Communication_Module. Missed: Collections_Workflow, Data_Warehouse, Reporting_Service.
  - Why it matters: Cross-team reporting, ensures aggregations light up properly.
- CASE07 [MISS] expected KYC_Service, Agent_Dashboard, Customer_Profile_Service; model predicted Notification_Engine. Hits: none. Extra: Notification_Engine. Missed: Agent_Dashboard, Customer_Profile_Service, KYC_Service.
  - Why it matters: Human-in-loop workflow with surfaces for agents.
- CASE08 [MISS] expected Dispute_Management, Collections_Workflow, Notification_Engine; model predicted Email_Communication_Module. Hits: none. Extra: Email_Communication_Module. Missed: Collections_Workflow, Dispute_Management, Notification_Engine.
  - Why it matters: Tests chaining of dispute + collections + alerts.
- CASE09 [PARTIAL] expected API_Gateway, Core_Banking_Integration, Transaction_History_Service; model predicted Core_Banking_Integration, Data_Warehouse, Transaction_History_Service. Hits: Core_Banking_Integration, Transaction_History_Service. Extra: Data_Warehouse. Missed: API_Gateway.
  - Why it matters: Latency/resiliency scenario referencing backend components.
- CASE10 [MISS] expected Customer_Portal_UI, Agent_Dashboard, Notification_Engine; model predicted Transaction_History_Service. Hits: none. Extra: Transaction_History_Service. Missed: Agent_Dashboard, Customer_Portal_UI, Notification_Engine.
  - Why it matters: Intentionally ambiguous scenario to highlight uncertainty/coverage gap.