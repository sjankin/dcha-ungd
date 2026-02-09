# DCHA-UNGD Lexicons

## Version
v1.0

## Overview
Climate (DC) and Health (DH) lexicons for DCHA candidate extraction, adapted from Lancet Countdown indicator practice.

## Lexicon Files
- `dc_terms.txt`: 33 climate/environmental domain terms
- `dh_terms.txt`: 28 health domain terms

## Matching Rules

### Tokenization
- Case-insensitive matching (casefold before comparison)
- Token-boundary matching is RECOMMENDED to avoid false matches (e.g., "health" should not match "healthy")

### Scope Rules

**DH (Health) Scope - HUMAN HEALTH ONLY:**
- Include: human health outcomes, diseases, mortality, morbidity
- EXCLUDE: "healthy economy", "healthy competition", "health of the oceans/planet", "healthy environment" (unless human health is explicitly referenced)

**DC (Climate) Scope:**
- Include: climate change, emissions, environmental degradation, climate action/policy
- Terms cover both impacts (climate change, global warming) and actions (decarbonization, net zero)

## Candidate Selection Rule
A sentence is a CANDIDATE if it contains:
- At least one DC term AND
- At least one DH term
- Both within the same sentence

## Provenance
Lexicons derived from Lancet Countdown indicator practice and adapted for UN General Debate corpus analysis.

## Known Limitations
- Multi-word terms may have boundary matching challenges
- "health" is a common term that may match non-human-health contexts (addressed via annotation guidelines)
- Some climate terms are general (e.g., "temperature") and may produce false positives
