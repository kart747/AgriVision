# AgriVision — Live Demo Script for Judges
**Matrix Fusion 4.0 AI Hackathon 2026**

---

## Demo Overview (10 minutes)

This script walks judges through all **6 stages** of the AgriVision pipeline in real-time.

**Time allocation:**
- Stage intro & demo: 8 min
- Q&A on architecture: 2 min

---

## Pre-Demo Checklist

Before judges arrive:
- [ ] Backend running: `python -m uvicorn backend.main:app --port 8000`
- [ ] Frontend ready: `http://localhost:5500`
- [ ] Test images ready in `TestData/` folder
- [ ] Evaluation report visible on desktop
- [ ] LLM prompt doc open in editor

---

## STAGE 01: Drone Capture
**Duration: 30 seconds**

### What We're Showing
"This is the input layer. In a real deployment, a drone captures field imagery with GPS metadata embedded. Since drone acquisition is out of hackathon scope, we've simulated this with real leaf photographs that include location data."

**Demo Action:**
1. Open `frontend/detect.html` in browser
2. Point to date/time capture (simulating metadata)
3. Click upload zone

**Say to judges:**
> "The image on screen — this was captured with GPS coordinates embedded in the EXIF metadata. When a farmer takes a photo with their smartphone on the field, the location is automatically captured. That's Stage 01."

---

## STAGE 02: Preprocessing
**Duration: 45 seconds**

### What We're Showing
"Once the image arrives, it goes through preprocessing — resizing, normalization, and quality checks."

**Demo Action:**
1. Select test image (e.g., `FREC_Scab_apple.jpg`)
2. Show upload preview
3. Point out blur detection happening in background

**Code Reference:** `backend/model/preprocess.py`

**Say:**
> "The system checks three things instantly:
> 1. Image quality — is it blurry? (Laplacian variance test)
> 2. Size normalization — all images resized to 224x224 for the CNN
> 3. Tensor conversion — converted to neural network format
> 
> If blur score is too high, we warn the farmer: 'Please capture a clearer photo.'
> This prevents garbage predictions."

**Show code snippet:** (Terminal window)
```python
# From preprocess.py
blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
if blur_score < BLUR_THRESHOLD:
    flag_low_quality()
```

---

## STAGE 03: CNN Model
**Duration: 1 minute 30 seconds**

### What We're Showing
"The core AI — a deep learning model trained to recognize diseases."

**Demo Action:**
1. Upload test image
2. Show model loading message
3. Click "Analyze" button
4. Show processing spinner

**Say:**
> "This is an EfficientNet B0 model — a state-of-the-art CNN pre-trained on ImageNet, then fine-tuned on 40,000+ leaf images from the PlantVillage dataset.
> 
> We chose EfficientNet because:
> • Computationally efficient — runs on a laptop without GPU
> • Proven accuracy on agricultural datasets
> • Fast inference (< 1 second per image)
> 
> The model produces:
> 1. Disease classification (which of 26 disease classes?)
> 2. Confidence score (0-100%)
> 3. Severity level (Low/Moderate/High)
> 4. Grad-CAM heatmap (explainability — shows WHERE in the leaf it detected disease)"

**Behind the scenes:**
- Model weights: `backend/model/weights/best_model.pth` (85 MB, loaded in ~2 seconds)
- Classes: 26 disease classes across Tomato/Apple/Grape + healthy variants

---

## STAGE 04: Prediction Output
**Duration: 45 seconds**

### What We're Showing
"The model's raw output — what did it see?"

**Wait for results to appear, then say:**

> "Look at the results panel on the right. The model reports:
> 
> • DISEASE: [shows detected disease, e.g., 'Apple Scab']
> • CONFIDENCE: [shows 87%]
> • SEVERITY: [Moderate/Low/High based on damage extent]
> • Confidence Bar: Visual indicator of model certainty
> 
> Notice the CONFIDENCE GATE we implemented:
> If confidence < 60%, we don't display a prediction. Instead, we ask the farmer
> to recapture a clearer image. This prevents false alarms.
> 
> The Grad-CAM heatmap shows EXPLAINABILITY — the red areas show where the model 
> 'looked' to make its decision. This builds farmer trust: 'I can see what the AI saw.'"

**Point to heatmap visualization**

---

## STAGE 05: LLM Module  
**Duration: 2 minutes**

### What We're Showing
"Now we move beyond classification to INTELLIGENCE. The disease label goes into an LLM that generates actionable recovery steps."

**Demo Action:**
1. Show "Loading Groq summary..." 
2. Wait for LLM response to appear
3. Point to the recommendation panel

**Say:**
> "Here's where AgriVision gets sophisticated. The disease prediction alone doesn't help a farmer. They need to know: WHAT DO I DO?
> 
> We pass the model output to Groq (a free LLM API):
> • Crop type
> • Disease detected
> • Confidence score
> • Farmer's LOCATION (from GPS)
> • Current MONTH
> 
> The LLM uses these inputs to generate location-specific, time-aware recommendations.
> 
> For example, if a farmer in Kerala detects Early Blight in April:
> • The LLM knows April = pre-monsoon (high humidity)
> • Knows Kerala = high rainfall region
> • Recommends: 'Spray by 4 PM before dew, reapply after rain'
> 
> If the same disease is detected in Rajasthan in October:
> • Knows October = post-monsoon (drier season)
> • Recommends: 'Moisture is low — focus on removing infected leaves'
> 
> Same disease, DIFFERENT recommendations based on context."

**Point to recommendation output:**
- Immediate Action
- Local Treatment
- Weather Advisory
- Recovery Time

**Technical details when asked:**
- Model: llama3-8b via Groq (free tier)
- Prompt: Prompt V2 (schema-locked JSON + confidence-aware rules)
- Temperature: 0.2-0.3 (factual, low-variance output)
- Context: Disease KB + location + month + confidence gate status
- Fallback: If LLM fails, disease-specific KB recommendations activate

---

## STAGE 06: Web App UI
**Duration: 1 minute**

### What We're Showing
"Everything above runs in a simple, farmer-friendly web interface."

**Demo Action:**
1. Show the full result card with all information
2. Point to navigation
3. Show responsiveness (if time permits)

**Say:**
> "The UX is intentionally simple. A farmer just needs to:
> 1. Upload a photo
> 2. See what disease it is
> 3. See what to do about it
> 
> No complex dashboards. No jargon. Just actionable intelligence.
> 
> We've also built in local storage — the system remembers past scans
> so farmers can track disease progression over time (Analytics dashboard
> is a bonus feature not in required scope, but shows data science thinking)."

**Show features:**
- [ ] Image upload (drag-drop)
- [ ] Crop selection (optional, auto-detects)
- [ ] GPS coordinates (auto-filled from EXIF)
- [ ] Results card with all metrics
- [ ] Grad-CAM visualization
- [ ] LLM recommendation panel

---

## ARCHITECTURE SUMMARY
**Say while showing diagram:**

```
┌──────────────────────────────────────────────────────────┐
│  STAGE 01: Farmer uploads photo with GPS metadata        │
└────────────────────┬─────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  STAGE 02: System checks quality (blur, size, format)    │
└────────────────────┬─────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  STAGE 03: CNN Model infers disease class & confidence   │
│            (EfficientNet B0, <1 second inference)        │
└────────────────────┬─────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  STAGE 04: Prediction displayed with Grad-CAM heatmap    │
│            (Explainability layer)                        │
└────────────────────┬─────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  STAGE 05: LLM generates location-aware recommendations  │
│            (Groq API + Disease KB + Location context)    │
└────────────────────┬─────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  STAGE 06: Results shown in simple farmer-facing UI      │
│            (Disease, Confidence, Treatment steps, etc.)  │
└──────────────────────────────────────────────────────────┘
```

---

## Expected Judge Questions & Answers

### Q: "How accurate is the model?"

**A:** 
> "The model achieves macro F1 score of 0.84 across all disease classes.
> That's 84% accuracy when averaging across classes (important for imbalanced data).
> 
> Per-class breakdown:
> • Tomato Healthy: 93% (high, as expected)
> • Tomato Early Blight: 85%
> • Apple Scab: 76% (challenging disease, requires clear symptoms)
> • Grape Black Measles: 81%
> 
> We have a full evaluation report with confusion matrix available.
> Plus, the 60% confidence gate means we REJECT predictions below that threshold,
> so what the farmer sees has higher practical accuracy."

**Reference:** `evaluation_results/evaluation_summary.txt`

---

### Q: "What if the LLM API goes down?"

**A:**
> "We have a fallback system. If Groq is unavailable, we switch to
> our local Disease Knowledge Base — a hand-curated JSON file with
> proven treatments for each disease-crop combination.
> 
> The farmer still gets recommendations, just from the disease-specific
> knowledge base instead of the LLM. We do not use a single hardcoded
> generic answer for all diseases.
> 
> This is critical for deployment — farmers in low-connectivity areas
> can STILL get advice without internet."

**Reference:** `backend/llm_validation/advisor.py` lines 45-90

---

### Q: "How does location context improve recommendations?"

**A:**
> "Different regions have different:
> • Rainfall patterns (monsoon vs. dry)
> • Humidity levels (affects fungal spread)
> • Available pesticides (regional regulations)
> • Growing seasons
> 
> For example, Early Blight in Kerala (humid, monsoon-prone) needs
> frequent reapplication because rain washes off spray. In Rajasthan
> (dry), frequency can be lower.
> 
> We extract location from image EXIF GPS, pass it to the LLM,
> and the LLM enriches recommendations with regional context.
> 
> This is the 'geo-aware' intelligence the hackathon asked for."

---

### Q: "Can this run offline / on edge?"

**A:**
> "Almost. Currently:
> • CNN model inference: FULLY LOCAL (no internet needed)
> • LLM recommendations: Requires Groq API call (network needed)
> 
> For full offline deployment, we'd need:
> • Local LLM (like Ollama with Mistral 7B)
> • Disease KB (already bundled locally)
> 
> This would require ~4-6 GB disk space and ~2GB RAM.
> For a 24-hour hackathon, we prioritized simplicity + Groq free tier.
> But the architecture supports offline if needed."

---

### Q: "Explain the Grad-CAM visualization."

**A:**
> "Grad-CAM (Gradient-weighted Class Activation Mapping) shows
> which pixels the model focused on when making its decision.
> 
> The red/hot areas = strong influence on disease prediction
> The blue/cool areas = less important
> 
> This is EXPLAINABILITY. Farmers can verify: 'Yes, I see the lesions
> where the AI is highlighting.' This builds trust vs. a black box.
> 
> If the AI highlights the wrong part of the leaf, farmers know
> the prediction might be unreliable."

**Reference:** `backend/model/gradcam.py`

---

### Q: "What's the confidence gate, and why 60%?"

**A:**
> "The confidence gate is a safety feature.
> 
> If the model is < 60% confident in its prediction, we don't show it.
> Instead, we ask the farmer: 'Please capture a clearer photo.'
> 
> Why 60%? Because below that, the disease classification is essentially
> a guess. Showing a guess as fact could lead to:
> • Wrong treatment (farmer treats what isn't there)
> • Unnecessary cost
> • Wasted time
> 
> It's better to ask for clarity than to mislead.
> This requirement was in the hackathon spec (Layer 2 validation)."

**Reference:** `backend/main.py` line 127

---

## Troubleshooting (If Demo Fails)

| Issue | Fix |
|-------|-----|
| Backend not running | Run: `python -m uvicorn backend.main:app --port 8000` |
| Model not loading | Check `model/weights/best_model.pth` exists (85 MB) |
| LLM slow/timeout | Groq might be rate-limited. Fallback to KB-only response. |
| Image not uploading | Check file size (<10 MB), format (JPG/PNG) |
| GPS not populating | Use sample GPS: 12.9352, 75.4030 (Mangalore) |

---

## Post-Demo Materials to Share With Judges

1. **evaluation_summary.txt** — F1, precision, recall per class
2. **LLM_PROMPT_DESIGN.md** — Full prompt engineering docs
3. **GitHub repo link** — Source code
4. **Model weights download link** — Easy deployment

---

## Key Talking Points

✅ **"We built what a farmer would actually use."**
> Simple interface. Real inference. Actionable advice. Not a research paper.

✅ **"All 6 stages work. No shortcuts."**
> CNN classification → LLM intelligence → Web UI. Full pipeline.

✅ **"We thought about failure modes."**
> Blur detection. Confidence gate. LLM fallback. GPS validation.

✅ **"Location + time context matters for agriculture."**
> Same disease, different treatments in Kerala vs. Rajasthan.

✅ **"Explainability builds farmer trust."**
> Grad-CAM heatmaps show where the model looked. Not a black box.

---

**Total script time: ~10 minutes + 2 min Q&A = 12 min presentation**

Good luck at the hackathon! 🚀
