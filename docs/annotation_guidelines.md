# DCHA-UNGD Annotation Guidelines

## Task Overview

Annotators label candidate sentences from UN General Debate speeches that contain within-sentence co-mentions of climate (DC) and health (DH) terms. Each candidate receives:

1. **Attribution label** (ATTRIB): Does the sentence assert a causal claim?
2. **Span annotations**: If ATTRIB=1, mark contiguous CAUSE and EFFECT spans
3. **Directed linkage type**: One of five categories

## Label Schema

### Level 1: Attribution Presence (ATTRIB)

- **ATTRIB=1**: The sentence contains an attributional causal claim connecting a cause and an effect
- **ATTRIB=0**: The sentence does not assert causation (e.g., lists topics, makes normative statements, or merely co-mentions climate and health)

### Level 1: Cause and Effect Spans

For ATTRIB=1 sentences, annotators provide:
- **CAUSE span**: The contiguous text expressing the cause
- **EFFECT span**: The contiguous text expressing the effect

Spans should be minimal but complete, capturing the core causal elements.

### Level 2: Directed Linkage Type

| Code | Label | Description |
|------|-------|-------------|
| 1 | **C2H_HARM** | Climate impacts cause health harms (e.g., "climate change spreads disease") |
| 2 | **C2H_COBEN** | Climate action produces health co-benefits (e.g., "clean energy improves air quality") |
| 3 | **H2C_JUST** | Health outcomes justify climate action (e.g., "for health reasons, we support climate action") |
| 4 | **OTHER_UNCLEAR** | Causal claim present but direction is ambiguous or does not fit above categories |
| 5 | **NO_CAUSAL_EXTRACTION** | No causal claim asserted |

### Derived Indicator: DCHA

DCHA=1 if all three conditions are met:
- ATTRIB=1
- The CAUSE span contains at least one DC (climate) term
- The EFFECT span contains at least one DH (health) term

## Decision Rules

### Listing vs. Attribution

Conjunctions like "and" or comma-separated enumerations indicate **listing** (NO_CAUSAL) unless accompanied by causal verbs or prepositional phrases.

| Sentence | Decision | Reason |
|----------|----------|--------|
| "Climate change and health are challenges we face." | NO_CAUSAL | Enumerative listing |
| "Climate change poses threats to health." | C2H_HARM | "poses threats to" implies causation |
| "Climate and health policies must work together." | NO_CAUSAL | Normative coordination, not causation |
| "We must protect health and fight climate change." | NO_CAUSAL | Parallel imperatives listing two goals |

### Hedged Language

Modal verbs ("may", "could", "might") or probability markers **do not negate attribution**. They hedge certainty but preserve the causal claim.

| Sentence | Decision | Reason |
|----------|----------|--------|
| "Climate change may contribute to the spread of disease." | C2H_HARM | Hedged but still asserts causal pathway |
| "Climate change could potentially affect health outcomes." | C2H_HARM | Modal does not negate attribution |

### Direction

The same concepts can appear in different causal directions. Annotators must identify which element is cause vs. effect.

| Sentence | Decision | Reason |
|----------|----------|--------|
| "Climate change causes health harm." | C2H_HARM | Climate = cause, health = effect |
| "For health reasons, we support climate action." | H2C_JUST | Health = justification, climate = action |

### Human vs. Other "Health"

References to "planetary health", "ocean health", or "ecosystem health" are coded as **OTHER_UNCLEAR** unless they explicitly connect to human health outcomes.

| Sentence | Decision | Reason |
|----------|----------|--------|
| "Health of the ocean is affected by climate." | OTHER | "Ocean health" is not human health |
| "Climate change threatens the health of coastal communities." | C2H_HARM | Human population health |

### Common Edge Cases

- **Policy-as-causation**: "The Organisation helped build a resilient Pacific" — code as OTHER if causal but not direct climate-to-health
- **Metaphorical causation**: "The world is crumbling under the weight of crises" — typically NO_CAUSAL unless a specific causal mechanism is identified
- **Nominal constructions**: "The impact of climate change on human health" — ATTRIB=1, C2H_HARM (nominal form implies causal relationship)
- **Appositive markers**: "Climate change, with its effect on disease spread, ..." — ATTRIB=1 if the appositive asserts a causal link

## Workflow

1. A RoBERTa model (trained on PolitiCAUSE) proposes initial ATTRIB labels and candidate spans
2. A trained annotator reviews and corrects the model output (post-editing)
3. Both pre-edit and post-edited labels are released

## Quality Assessment

Inter-annotator agreement on a 100-candidate subsample:
- Cohen's kappa = 0.42 (moderate agreement, consistent with task difficulty)
- Raw agreement = 71%
- On 35 agreed ATTRIB=1 cases: cause span F1 = 0.80, effect span F1 = 0.79
