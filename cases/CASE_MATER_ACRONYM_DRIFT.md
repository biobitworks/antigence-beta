# CASE STUDY: MATER Acronym Drift

## Error Classification
**Type**: Semantic Drift / Hallucinated Terminology Replacement  
**Severity**: HIGH - Core project identity corrupted  
**Date Discovered**: 2026-01-13  
**Session**: f566244c-6ce3-4777-abd3-045b434e9d4d  

---

## Summary

An AI agent session silently changed the fundamental project acronym from the user-defined original to a hallucinated alternative, propagating the error across 20+ documents without user consent.

---

## The Error

| Attribute           | Original (Correct)                                          | AI-Modified (Wrong)                                                |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ |
| **MATER Expansion** | Maternal **Analog Timer** for Epigenetics and Reprogramming | Maternal **Aggregate Transmission Establishes** Reproductive aging |
| **Key Difference**  | "Analog Timer" + "Reprogramming"                            | "Aggregate Transmission" + "Establishes"                           |
| **Semantic Shift**  | Timer/clock metaphor + therapeutic (Reprogramming)          | Disease transmission + declaration                                 |

---

## Evidence Trail

### Where Original Was Preserved (Gemini Brain Files)

```
/Users/byron/.gemini/antigravity/brain/f01920e9-9125-45c1-a758-31c03c0e5413/walkthrough.md.resolved.8
  Line 54: "MATER: Maternal Analog Timer for Epigenetics and Reprogramming"

/Users/byron/.gemini/antigravity/code_tracker/active/MATER_e97b2f1180ca007894a0efb1dfd8e31d5e2238b5/c378660136105957d33de95e1947963c_VALIDATION_PROMPT_FOR_CHATGPT_2026-01-12.md
  Line 3: "MATER (Maternal Analog Timer for Epigenetics and Reprogramming)"

/Users/byron/.gemini/antigravity/code_tracker/active/MATER_ee60c87fcf2a28a948450660ec9c2ed18aed6893/062980c6e15bb45dc60f19d17ac5229a_00_Index.md
  Line 3: "Project: Maternal Analog Timer for Epigenetics and Reprogramming"
```

### Where Error Propagated (Main Project Files)

```
/Users/byron/projects/active_projects/MATER/README.md
/Users/byron/projects/active_projects/MATER/theory/MATER_naming_framework.md (line 8)
/Users/byron/projects/active_projects/MATER/docs/SESSION_SUMMARY_2026-01-11.md (line 125)
/Users/byron/projects/active_projects/MATER/docs/JOURNAL_2026-01-13.md (line 36)
... and ~20 more files
```

---

## Root Cause Analysis

### 1. No Canonical Definition Lock
The project lacked a single source of truth for the acronym that agents must read before modifying naming.

### 2. No Diff Review for Critical Terms
When the AI updated `MATER_naming_framework.md`, the change to line 8 was not flagged for user review.

### 3. Insufficient Session Context Persistence
The correct definition existed in previous session brain files, but was not loaded into new sessions.

### 4. Hallucination Confidence
The AI generated a plausible-sounding alternative and stated it was "formalized" without user confirmation.

---

## Detection Method

1. User noticed inconsistency when reviewing grant application
2. User asked: "When did you hallucinate this change?"
3. Agent searched chat history in `.gemini/antigravity/brain/`
4. Original definition found in resolved artifact files

---

## Remediation Required

1. **Immediate**: Fix all MATER project files to use correct expansion
2. **Systemic**: Implement canonical term locks in project context files
3. **Verification**: Add pre-flight checks for critical term consistency

---

## Lessons for Antigence

### Detection Rules Needed

```yaml
rule: canonical_term_drift
  trigger: modification of defined acronym/term expansion
  action: require_explicit_user_confirmation
  examples:
    - "MATER" expansion
    - Project names
    - Hypothesis names

rule: cross_session_consistency
  trigger: new session starts
  action: load_canonical_definitions_from_context_file
  verify: current_usage_matches_canonical
```

### Context File Requirements

Every project should have a `CANONICAL_TERMS.md` or entry in `context.json`:

```json
{
  "canonical_terms": {
    "MATER": "Maternal Analog Timer for Epigenetics and Reprogramming",
    "CytoClock": "Cytoplasmic Inheritance Timer",
    ...
  },
  "term_lock_level": "user_confirmation_required"
}
```

### Agent Pre-Flight Checklist

Before modifying any document:
- [ ] Load canonical terms from project context
- [ ] Check if any canonical terms are being modified
- [ ] If yes, halt and request explicit user confirmation
- [ ] Log all term modifications with session ID and timestamp

---

## Related Error Patterns

- **Synonym Substitution**: AI replaces domain-specific term with "equivalent" that changes meaning
- **Expansion Drift**: Acronym expansions modified to be "more accurate"
- **Naming "Improvements"**: AI decides original name is suboptimal and silently "fixes" it

---

## Files Affected (To Be Corrected)

```
/Users/byron/projects/active_projects/MATER/README.md
/Users/byron/projects/active_projects/MATER/theory/MATER_naming_framework.md
/Users/byron/projects/active_projects/MATER/docs/SESSION_SUMMARY_2026-01-11.md
/Users/byron/projects/active_projects/MATER/docs/JOURNAL_2026-01-13.md
/Users/byron/projects/active_projects/MATER/docs/MATER_PUBLICATION_FRAMEWORK.md
/Users/byron/projects/active_projects/MATER/obsidian/00_Index.md
... (20+ files total)
```

---

*Case documented by Antigence TRAITS verification system*  
*Discovery timestamp: 2026-01-13T00:19:17-08:00*
