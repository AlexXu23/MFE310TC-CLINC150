# Tool assistance and team review

ChatGPT/Codex assisted with implementation, experiment execution, statistical checks, error-case interpretation and document drafting. The package includes source code, configurations, predictions and independent numerical checks supporting the reported results.

## Review records

- Numerical verification covers source-row alignment, saved logits, classification metrics, rejection decisions, duplicate sensitivity and the three-seed summaries. The transfer checks in `environment/` were executed against this portable copy.
- Qualitative interpretations are stored in `analysis/qualitative_annotations.json` and `outputs/error_analysis/reviewed_cases_ai_assisted.csv`. The CSV includes `human_confirmed=false`; team review should add reviewer identity, date and any amended interpretation in a working copy.
- Personal-contribution statements should identify each member's actual implementation, reproduction, analysis and presentation work. This handoff does not assign or certify individual contributions.

## Suggested declaration structure

Record the tool name, assisted activities, artifacts independently checked by the team, and how the final interpretations and submission text were reviewed. The final declaration should match the group's actual review process and the coursework requirements.

## Working convention

Keep the supplied archive as the reference snapshot. Use new output directories for new training/evaluation and record any changes to data, model settings, thresholds or inference precision. Group discussions can cite exact CSV rows or case ids from this snapshot.
