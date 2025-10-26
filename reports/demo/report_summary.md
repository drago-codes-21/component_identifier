# Component Identifier – Validation Snapshot

## Key Takeaways
- Reviewed 300 validation stories that mention 20 different components.
- The model picked the exact right combination of components 100.0% of the time.
- Looking across all component tags, it was correct 100.0% of the time when it raised a component (precision) and found 100.0% of the components that should be flagged (recall).
- On average each story touches 3.2 components, so the near-perfect F1 score (100.0%) means stakeholders can trust the recommendations.
- Lower validation loss (the recorded checkpoints) across epochs shows the model keeps learning without overfitting.

## Friendly Metric Definitions
- **Precision**: When the model says a component is impacted, how often it is actually impacted.
- **Recall**: Out of the components that truly need attention, how many the model successfully finds.
- **F1 Score**: Single number that balances precision and recall; helpful when both matter.
- **Exact Match**: Stories where the model predicted the full set of impacted components with no misses.
- **Loss**: Training objective number; lower is better because it means the predictions align with reality.

## Visuals
- Validation F1: `reports\demo\figures\f1_over_epochs.png`
- Validation Loss: `reports\demo\figures\loss_over_epochs.png`
- Component coverage (top 10): `reports\demo\figures\component_support.png`