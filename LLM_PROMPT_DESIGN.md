# LLM Prompt Engineering Documentation (V2)
AgriVision - Matrix Fusion 4.0 AI Hackathon 2026

---

## 1. Objective

The LLM layer converts CNN predictions into **actionable, location-aware farmer guidance**.

Input from vision stack:
- crop
- disease
- confidence
- severity
- location (GPS / user input)
- time context (month)

Output to UI:
- structured recommendations with safety-aware wording
- consistent JSON fields for reliable parsing
- fallback guidance if LLM is unavailable

---

## 2. Design Goals For Problem Statement

Prompt V2 is tuned for judge criteria:
- practical recommendations, not generic chatbot text
- location + seasonal adaptation in advice
- confidence-aware cautioning before chemical action
- predictable machine-readable output
- anti-hallucination constraints

---

## 3. Runtime Paths

### A) Live predict path (used by /predict)
- File: backend/llm/advisor.py
- Model: Groq llama3-8b-8192
- Output keys:
  - immediate_action
  - local_treatment
  - weather_warning
- Fallback: disease-specific knowledge-base response

### B) Judge/extended recommendation path (used by /generate-recommendation)
- Files:
  - backend/llm_validation/prompts.py
  - backend/llm_validation/advisor.py
- Model config: from backend/llm_validation/config.py
- Output schema:
  - source
  - crop
  - disease
  - summary
  - organic_treatment
  - chemical_treatment
  - recovery_time
  - preventive_measures
  - warnings
  - notes

---

## 4. Prompt V2 - System Prompt Strategy

System prompt enforces:
- JSON-only output
- exact schema keys
- urgency handling for Severe/Critical severity
- re-validation warning when confidence is low
- region-aware recommendations using location + season context
- no invented chemicals, regulations, or unsupported claims

This prevents:
- markdown or prose leakage
- hallucinated products/dosages
- unstable formatting in UI

---

## 5. Prompt V2 - User Prompt Template

User prompt is deterministic and sectioned:
- TASK
- PREDICTION_CONTEXT
- KNOWLEDGE_BASE_SIGNALS
- RESPONSE_REQUIREMENTS

Important logic embedded in prompt context:
- confidence gate status derived from threshold (0.60)
- recovery baseline from knowledge base
- instructions for farmer-friendly language
- warnings required when confidence is low

---

## 6. Safety And Reliability Controls

### Confidence-aware behavior
If confidence is below threshold:
- prompt instructs warning before chemical spray
- encourages clearer image recapture and validation

### Response normalization
In advisor post-processing, fields are normalized so missing arrays do not break downstream parsing.

### Validation
Prompt response validation checks required schema fields before accepting output.

### Fallback (no hardcoded generic output path)
If Groq API key is missing or API fails:
- system returns knowledge-base-backed disease-specific recommendations
- unknown disease still returns safe generic guidance + consult expert warning

---

## 7. Why This Is Better Than V1

V1 limitations:
- broad generic instructions
- weaker schema enforcement
- limited confidence-gated behavior

V2 improvements:
- stronger contract and deterministic structure
- explicit safety and uncertainty handling
- tighter grounding on knowledge-base context
- better consistency for judge live demos

---

## 8. Judge Talking Points (Use Verbatim)

1. "Our prompt is not just descriptive. It is constraint-driven and schema-driven, so the UI always receives structured, parseable output."
2. "We explicitly model uncertainty: if confidence is low, the recommendation asks for re-capture/verification before chemical spray."
3. "We combine model prediction + location + season + curated disease knowledge to produce region-aware actions instead of generic chatbot advice."
4. "If LLM fails, recommendation quality degrades gracefully to disease-specific knowledge-base guidance, not random hardcoded text."

---

## 9. Example Prompt V2 Context (Simplified)

```text
PREDICTION_CONTEXT:
- crop: Tomato
- disease: Tomato Early Blight
- confidence: 58.0%
- severity: Moderate
- location: Mangalore, Karnataka, India
- time_context: April
- confidence_gate_status: LOW (threshold 0.60)

KNOWLEDGE_BASE_SIGNALS:
- symptoms:
  - Brown concentric lesions
  - Yellow halo on leaves
- notes:
  - Spreads faster under high humidity

RESPONSE_REQUIREMENTS:
- Return JSON only with required keys.
- Include warning about re-capture/verification before chemical spray.
```

---

## 10. File Map

- backend/llm/advisor.py
- backend/llm_validation/prompts.py
- backend/llm_validation/advisor.py
- backend/llm_validation/config.py
- backend/llm_validation/data/disease_knowledge.json

---

## 11. Versioning

Prompt template versions in metadata:
- system_prompt_version: 2.0
- user_prompt_template_version: 2.0

This is the version evaluated in current submission package.

---

## 12. Sample Prompt-Response Pairs

### Pair A: High Confidence Case (Tomato Early Blight)

Sample Prompt (condensed):

```text
PREDICTION_CONTEXT:
- crop: Tomato
- disease: Tomato Early Blight
- confidence: 0.89
- severity: Moderate
- location: Mangalore, Karnataka, India
- time_context: April
- confidence_gate_status: PASS (threshold 0.60)

KNOWLEDGE_BASE_SIGNALS:
- symptoms: Brown concentric lesions, yellow leaf halo
- notes: Spreads quickly in high humidity
- expected_recovery_days: 21

RESPONSE_REQUIREMENTS:
- JSON only
- include treatment options and preventive measures
```

Sample Response:

```json
{
  "immediate_action": "Remove visibly infected lower leaves within 24 hours and isolate plant waste.",
  "local_treatment": "Start fungicidal schedule immediately and keep canopy dry with morning irrigation only.",
  "weather_warning": "High humidity in coastal Karnataka can accelerate spread; avoid evening watering and spraying before rain.",
  "organic_treatment": [
    "Spray neem oil-based bio-fungicide on affected area every 5-7 days.",
    "Use copper-based organic fungicide as per label direction."
  ],
  "chemical_treatment": [
    "Apply Mancozeb-based fungicide as per labeled dose.",
    "Rotate with Chlorothalonil-class option to reduce resistance risk."
  ],
  "recovery_time": "14-21 days with timely intervention",
  "preventive_measures": [
    "Maintain spacing for airflow between plants.",
    "Remove fallen infected debris weekly.",
    "Monitor new lesions every 2 days during humid weather."
  ]
}
```

### Pair B: Low Confidence Case (Apple Scab)

Sample Prompt (condensed):

```text
PREDICTION_CONTEXT:
- crop: Apple
- disease: Apple Scab
- confidence: 0.58
- severity: Mild
- location: Himachal Pradesh, India
- time_context: March
- confidence_gate_status: LOW (threshold 0.60)

KNOWLEDGE_BASE_SIGNALS:
- symptoms: Olive-brown lesions, leaf spotting
- notes: Visual overlap with powdery mildew can occur
- expected_recovery_days: 21

RESPONSE_REQUIREMENTS:
- JSON only
- include caution for low-confidence prediction before chemical spray
```

Sample Response:

```json
{
  "immediate_action": "Capture 2-3 clearer close-up photos from different leaves before final spray decision.",
  "local_treatment": "Prune heavily affected leaves and improve canopy airflow while awaiting verification.",
  "weather_warning": "Avoid spray in windy or rainy windows; use calm, dry morning conditions after verification.",
  "organic_treatment": [
    "Use sulfur-based organic spray after confirmation.",
    "Collect and dispose infected fallen leaves away from orchard."
  ],
  "chemical_treatment": [
    "Use approved scab fungicide only after re-validation confirms diagnosis."
  ],
  "recovery_time": "~21 days with early and confirmed intervention",
  "preventive_measures": [
    "Perform weekly orchard scouting during early season.",
    "Avoid prolonged leaf wetness by spacing and pruning.",
    "Maintain sanitation of pruned material and tools."
  ]
}
```
