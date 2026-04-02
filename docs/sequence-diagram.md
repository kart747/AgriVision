# AgriVision Sequence Diagrams

## Single Image Prediction Flow

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Frontend as 🌐 Frontend (detect.html)
    participant Backend as ⚡ FastAPI Backend
    participant Validator as 🛡️ File Validator
    participant Preprocessor as 🔄 Image Preprocessor
    participant Model as 🤖 EfficientNet-B0
    participant PostProc as 📈 Post-Processor
    participant Validation as ✔️ Validation Module
    participant Advisory as 💡 Advisory Engine
    participant LLM as 🧠 Groq LLM
    participant KB as 📚 Knowledge Base

    Note over User,LLM: Step 1: User uploads image
    User->>Frontend: Upload leaf image
    Frontend->>Frontend: Display image preview
    
    Note over User,LLM: Step 2: Send to backend
    Frontend->>Backend: POST /predict<br/>{image, crop_hint, location}
    
    Note over User,LLM: Step 3: File validation
    Backend->>Validator: Validate file type/size
    Validator-->>Backend: ✅ Valid or ❌ Error
    
    alt File Invalid
        Backend-->>Frontend: 400 Bad Request
        Frontend->>User: Show error message
    else File Valid
        Note over User,LLM: Step 4: Preprocessing
        Backend->>Preprocessor: Resize to 224×224
        Backend->>Preprocessor: Normalize (ImageNet stats)
        Backend->>Preprocessor: Convert to tensor
        Preprocessor-->>Backend: image_tensor, blur_score
        
        Note over User,LLM: Step 5: Model inference
        Backend->>Model: Forward pass (image_tensor)
        Model-->>Backend: logits (17 classes)
        
        Note over User,LLM: Step 6: Post-processing
        Backend->>PostProc: Apply softmax
        PostProc-->>Backend: probabilities
        Backend->>PostProc: argmax → class_index
        PostProc-->>Backend: disease_name, confidence %
        
        Note over User,LLM: Step 7: Grad-CAM generation
        Backend->>PostProc: Generate Grad-CAM heatmap
        PostProc-->>Backend: cam_image (base64)
        
        Note over User,LLM: Step 8: Validation checks
        Backend->>Validation: Check confidence >= 60%
        
        alt Confidence < 60%
            Validation-->>Backend: ❌ Low confidence
            Backend-->>Frontend: 422 Unprocessable Entity
            Frontend->>User: "Please capture a clearer image"
        else Confidence >= 60%
            Validation-->>Backend: ✅ Pass
            
            Note over User,LLM: Step 9: Validate image quality
            Backend->>Validation: Check blur_score threshold
            Validation-->>Backend: ✅ Quality OK or ⚠️ Warning
            
            Note over User,LLM: Step 10: Validate crop hint (if provided)
            alt crop_hint provided
                Backend->>Validation: Compare crop_hint vs detected
                alt Mismatch
                    Validation-->>Backend: ❌ Crop mismatch
                    Backend-->>Frontend: 422 Crop hint error
                    Frontend->>User: Show mismatch error
                else Match
                    Validation-->>Backend: ✅ Match
                end
            end
            
            Note over User,LLM: Step 11: Advisory generation
            Backend->>Advisory: Get recommendation request
            
            Note over User,LLM: Step 12: Try LLM first
            Advisory->>LLM: Request treatment advice<br/>{crop, disease, location, month}
            
            alt GROQ_API_KEY available
                LLM-->>Advisory: ✅ Recommendation (LLM response)
            else No API key
                Advisory->>KB: Get fallback recommendation
                KB-->>Advisory: ✅ Default recommendation
            end
            
            Note over User,LLM: Step 13: Build response
            Advisory-->>Backend: {disease, confidence, severity, recommendation}
            Backend-->>Frontend: 200 Success Response
            
            Note over User,LLM: Step 14: Display results
            Frontend->>User: Show disease, confidence, severity,<br/>recommendations, Grad-CAM heatmap
        end
    end
```

---

## Drone Scan (Batch Processing) Flow

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Frontend as 🌐 Frontend
    participant Backend as ⚡ FastAPI Backend
    participant Model as 🤖 EfficientNet-B0
    participant BatchProc as 📦 Batch Processor
    participant Aggregator as 📊 Result Aggregator

    Note over User,Aggregator: Step 1: User uploads multiple images
    User->>Frontend: Select multiple leaf images
    Frontend->>Backend: POST /drone-scan<br/>{images[]}
    
    Note over User,Aggregator: Step 2: Process each image
    Backend->>BatchProc: For each image in images[]
    
    loop For each image
        BatchProc->>Model: Run inference
        Model-->>BatchProc: disease, confidence
        BatchProc->>Aggregator: Add to results
    end
    
    Note over User,Aggregator: Step 3: Aggregate results
    Aggregator-->>Backend: {total_scanned, infected_count, healthy_count, primary_threat, per_leaf_results}
    
    Note over User,Aggregator: Step 4: Return batch results
    Backend-->>Frontend: Batch response
    Frontend->>User: Show summary:<br/>- X leaves scanned<br/>- Y infected, Z healthy<br/>- Primary threat: Disease Name
```

---

## Error Handling Flow

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Frontend as 🌐 Frontend
    participant Backend as ⚡ FastAPI

    Note over User,Backend: Error Case 1: File validation failed
    User->>Frontend: Upload invalid file
    Frontend->>Backend: POST /predict
    Backend-->>Frontend: 400 Bad Request<br/>"File type must be image"
    Frontend->>User: "Please upload a valid image (PNG, JPEG, WebP)"
    
    Note over User,Backend: Error Case 2: Model not loaded
    User->>Frontend: Try to predict
    Backend-->>Frontend: 503 Service Unavailable<br/>"Model not loaded"
    Frontend->>User: "Service temporarily unavailable"
    
    Note over User,Backend: Error Case 3: Low confidence
    User->>Frontend: Upload blurry image
    Backend-->>Frontend: 422 Unprocessable<br/>"Confidence below 60%"
    Frontend->>User: "Please capture a clearer leaf image"
    
    Note over User,Backend: Error Case 4: Crop mismatch
    User->>Frontend: Select "Apple", upload grape leaf
    Backend-->>Frontend: 422 Unprocessable<br/>"Crop hint mismatch"
    Frontend->>User: "Detected different crop. Retry without crop selection or upload correct leaf."
    
    Note over User,Backend: Error Case 5: LLM API failure
    User->>Frontend: Make prediction request
    Backend->>Backend: LLM API timeout/error
    Backend->>Backend: Use fallback recommendations
    Backend-->>Frontend: 200 Success (with default recommendations)
    Frontend->>User: Show results with ⚠️ "Offline mode" indicator
```

---

## Validation Check Sequence

```mermaid
sequenceDiagram
    participant Backend as ⚡ Backend
    participant Validation as ✔️ Validation Module

    Note over Backend,Validation: Confidence Validation
    Backend->>Validation: Check confidence >= 60%
    alt confidence < 60
        Validation-->>Backend: ❌ REJECT<br/>"Low confidence prediction"
    else confidence >= 60
        Validation-->>Backend: ✅ PASS
    end

    Note over Backend,Validation: Image Quality Check
    Backend->>Validation: Check blur_score threshold
    alt blur_score < threshold
        Validation-->>Backend: ⚠️ WARNING<br/>"Image may be blurry"
    else blur_score >= threshold
        Validation-->>Backend: ✅ PASS
    end

    Note over Backend,Validation: GPS Validation (if provided)
    Backend->>Validation: Validate latitude/longitude
    alt coordinates invalid
        Validation-->>Backend: ⚠️ WARNING<br/>"Invalid GPS coordinates"
    else coordinates valid
        Validation-->>Backend: ✅ PASS
    end

    Note over Backend,Validation: Crop Hint Validation (if provided)
    Backend->>Validation: Compare crop_hint vs detected crop
    alt crop_hint != detected_crop
        Validation-->>Backend: ❌ REJECT<br/>"Crop hint mismatch"
    else crop_hint == detected_crop
        Validation-->>Backend: ✅ PASS
    end

    Note over Backend,Validation: File Validation
    Backend->>Validation: Check file type and size
    alt invalid file
        Validation-->>Backend: ❌ REJECT<br/>"Invalid file type or size"
    else valid file
        Validation-->>Backend: ✅ PASS
    end
```

---

## Key Timing Notes

| Step | Approximate Time |
|------|------------------|
| File upload | Depends on network |
| Preprocessing | ~50ms |
| Model inference | ~100-200ms (GPU), ~500ms (CPU) |
| Grad-CAM generation | ~100ms |
| LLM API call | ~1-3s (if available) |
| **Total** | **~1-4 seconds** |

---

## Retry Logic

```mermaid
flowchart LR
    A[First Request] --> B{Success?}
    B -->|Yes| C[Return Result]
    B -->|No| D{Crop Mismatch Error?}
    D -->|Yes| E[Retry without crop_hint]
    E --> F{Success?}
    F -->|Yes| C
    F -->|No| G[Show Error]
    D -->|No| G
```

The frontend has built-in retry logic:
1. If crop mismatch error → retry without crop_hint
2. If other error → show error to user