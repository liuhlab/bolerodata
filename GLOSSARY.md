# bolerodata

The registry of datasets, trained models and evaluation records behind the Bolero paper, and
where each one lives.

## Language

**Tutorial input**:
A file a tutorial loads that it does not produce itself. Every tutorial input is downloadable
by the public.
_Avoid_: initial input, tutorial data

**Tutorial output**:
A file a tutorial produces by running. Never published; readers regenerate it.

**Published stand-in**:
A published model or file that a tutorial loads in place of the lab-only one the lab run used.
_Avoid_: substitute, replacement

**Bolero-AP1-Score**:
The paper's AP-1 activity-score model: Bolero conditioned on a per-pseudobulk AP-1 activity
score as well as cell state. Its model-zoo key is `Bolero10M+AP1+260104`.
_Avoid_: AP1 model, score model (ambiguous with the other motif score models)
