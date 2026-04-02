# AgriVision System Architecture

```mermaid
flowchart TB
    %% Colors
    classDef client fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef api fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef ml fill:#fff3e0,stroke:#f57c00,color:#e65100
    classDef validation fill:#fce4ec,stroke:#c2185b,color:#880e4f
    classDef advisory fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef data fill:#e0f7fa,stroke:#0097a7,color:#006064
    classDef training fill:#f1f8e9,stroke:#689f38,color:#33691e
    classDef endpoint fill:#fff8e1,stroke:#ff8f00,color:#ff6f00

    subgraph Client["🎨 Client Layer"]
        direction TB
        WC[🌐 Web Clients]
        MC[📱 Mobile Apps]
        
        subgraph FrontendFiles
            UI1[agrigo.html]
            UI2[index.html]
            UI3[detect.html]
            UI4[test_ui.html]
        end
        
        WC --> UI1
        WC --> UI2
        MC --> UI3
        UI1 --> FrontendFiles
        UI2 --> FrontendFiles
        UI3 --> FrontendFiles
        UI4 --> FrontendFiles
    end
    class Client client

    subgraph API["🚀 API Gateway Layer"]
        direction TB
        FastAPI[⚡ FastAPI Server]
        Uvicorn[🐳 Uvicorn]
        CORS[CORS Middleware]
        Endpoints
        
        subgraph API_Endpoints["🔌 API Endpoints"]
            POSTPredict[POST /predict<br/>Single Image]
            POSTDrone[POST /drone-scan<br/>Batch Analysis]
            POSTValidate[POST /validate]
            POSTAdvice[POST /advice]
            GETHealth[GET /health]
            GETClasses[GET /classes]
        end
        
        FastAPI --> Uvicorn
        CORS --> FastAPI
        FastAPI --> Endpoints
        Endpoints --> API_Endpoints
    end
    class API api

    subgraph Pipeline["⚙️ Processing Pipeline"]
        direction LR
        Upload[📤 File Upload]
        FileVal[✅ File Validation]
        Preprocess[🔄 Image Preprocessing]
        Resize[📏 Resize/Normalize]
        Tensor[🔢 Tensor Conversion]
        
        Upload --> FileVal
        FileVal --> Preprocess
        Preprocess --> Resize
        Resize --> Tensor
    end
    class Pipeline api

    subgraph ML["🤖 ML Model Layer"]
        direction TB
        EfficientNet[EfficientNet-B0<br/>Fine-tuned Model]
        DiseaseClasses[📋 14 Disease Classes]
        
        subgraph Classes["Disease Classes"]
            Tomato[🍅 Tomato<br/>9 diseases + healthy]
            Apple[🍎 Apple<br/>2 diseases + healthy]
            Grape[🍇 Grape<br/>1 disease + healthy]
        end
        
        subgraph Inference["🔍 Inference"]
            Input[📷 Image Tensor<br/>224×224×3]
            Forward[Forward Pass]
            Output[Softmax Output]
            ArgMax[argmax]
            Confidence[📊 Confidence %]
        end
        
        EfficientNet --> DiseaseClasses
        DiseaseClasses --> Classes
        Input --> Forward
        Forward --> Output
        Output --> ArgMax
        Output --> Confidence
        Classes --> ArgMax
    end
    class ML ml

    subgraph PostProcess["📈 Post-Processing Layer"]
        direction LR
        GradCAM[🎯 GradCAM<br/>Visualization]
        Severity[📉 Severity<br/>Calculation]
        BlurDetect[👁️ Blur Detection]
        
        BlurDetect -.->|"Laplacian<br/>Variance"| BlurScore[Blur Score]
    end
    class PostProcess ml

    subgraph Validation["🛡️ Validation Module"]
        direction TB
        ConfVal[📊 Confidence Check<br/>Threshold: 60%]
        BlurVal[👁️ Image Quality<br/>Check]
        GPSVal[📍 GPS Validation]
        CropVal[🌾 Crop Hint<br/>Validation]
        BusinessRules[📋 Business Rules]
        
        ConfVal -->|"Pass/Fail"| ValidationResult
        BlurVal --> ValidationResult
        GPSVal --> ValidationResult
        CropVal --> ValidationResult
        BusinessRules --> ValidationResult
    end
    class Validation validation

    subgraph Advisory["💡 Advisory Engine"]
        direction TB
        LLMEngine[🧠 LLM Engine<br/>Groq API]
        KnowledgeBase[📚 Knowledge Base]
        Fallback[🔄 Fallback System]
        
        subgraph RecommendationTypes["📋 Recommendation Types"]
            RecImmediate[🚨 Immediate Action]
            RecLocal[💊 Local Treatment]
            RecWeather[🌤️ Weather Warning]
        end
        
        LLMEngine --> RecImmediate
        LLMEngine --> RecLocal
        LLMEngine --> RecWeather
        
        KnowledgeBase --> Fallback
        Fallback --> RecImmediate
        Fallback --> RecLocal
        Fallback --> RecWeather
        
        LLMEngine -.->|"GROQ_API_KEY"| GroqCloud[☁️ Groq Cloud]
    end
    class Advisory advisory

    subgraph DataSources["💾 Data Sources"]
        disease_knowledge[disease_knowledge.json<br/>Disease Info]
        farm_regions[farm_regions.json<br/>Regional Data]
        sample_cases[sample_cases.json<br/>Example Cases]
        model_weights[best_model.pth<br/>Model Weights]
        class_names[class_names.json<br/>Class Labels]
    end
    class DataSources data

    subgraph Training["🎓 Training Pipeline"]
        direction TB
        PlantVillage[🌱 PlantVillage<br/>Dataset]
        PlantDoc[📷 PlantDoc<br/>Dataset]
        DataAugmentation[🔄 Data Augmentation<br/>Flip/Rotate/ColorJitter]
        TrainingScript[📝 train_unified.py<br/>Training Script]
        Evaluation[📊 Model Evaluation<br/>Accuracy: 94.65%]
        
        PlantVillage --> DataAugmentation
        PlantDoc --> DataAugmentation
        DataAugmentation --> TrainingScript
        TrainingScript --> Evaluation
        Evaluation --> model_weights
    end
    class Training training

    %% Connections
    Client --> API
    API --> Pipeline
    Pipeline --> ML
    ML --> PostProcess
    PostProcess --> Validation
    Validation --> Advisory
    Advisory --> API
    
    model_weights --> ML
    class_names --> ML
    KnowledgeBase --> Advisory
    disease_knowledge -.-> KnowledgeBase
    farm_regions -.-> KnowledgeBase
    sample_cases -.-> KnowledgeBase
    PlantVillage -.-> Training
```

---

## Component Descriptions

### 🎨 Client Layer
- **agrigo.html**: Landing page with product information
- **index.html**: Main dashboard with features overview
- **detect.html**: Disease detection interface (primary)
- **test_ui.html**: API testing UI for developers

### 🚀 API Gateway Layer
- **FastAPI**: Python web framework for REST APIs
- **Uvicorn**: ASGI server for running FastAPI
- **CORS**: Cross-origin resource sharing enabled

### ⚙️ Processing Pipeline
1. **File Upload**: Accepts image files (PNG, JPEG, WebP)
2. **File Validation**: Checks file type, size limits
3. **Preprocessing**: Resizes to 224×224, normalizes
4. **Tensor Conversion**: Converts to PyTorch tensor

### 🤖 ML Model Layer
- **EfficientNet-B0**: Lightweight CNN, pre-trained on ImageNet
- **14 Classes**: 3 Apple + 2 Grape + 9 Tomato diseases (including healthy)
- **Training**: Fine-tuned on PlantVillage + PlantDoc dataset

### 📈 Post-Processing Layer
- **Grad-CAM**: Visual heatmap showing disease-affected regions
- **Severity Score**: 0-100 damage assessment
- **Blur Detection**: Checks image quality via Laplacian variance

### 🛡️ Validation Module
- **Confidence Threshold**: Rejects predictions below 60%
- **Image Quality**: Flags blurry images
- **GPS Validation**: Validates coordinates for location-aware advice
- **Crop Hint**: Optional crop selection validation

### 💡 Advisory Engine
- **Groq LLM**: Uses llama3-8b-8192 model for recommendations
- **Knowledge Base**: JSON files with disease information
- **Fallback**: Default recommendations if API unavailable

### 💾 Data Sources
- `disease_knowledge.json`: Disease symptoms and treatments
- `farm_regions.json`: Regional farming data
- `sample_cases.json`: Example prediction cases
- `best_model.pth`: Trained model weights
- `class_names.json`: 14 class labels

### 🎓 Training Pipeline
- **Dataset**: PlantVillage + PlantDoc (~17,000 balanced images after merging)
- **Augmentation**: Horizontal/Vertical flip, rotation, color jitter
- **Training Script**: train_unified.py with frozen + unfrozen epochs
- **Evaluation**: 94.65% accuracy on test set

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single image disease detection |
| `/drone-scan` | POST | Batch analysis for multiple images |
| `/validate` | POST | Validate prediction confidence |
| `/advice` | POST | Get treatment recommendations |
| `/health` | GET | Check server status and model |
| `/classes` | GET | List supported disease classes |

---

## Dependencies

### Python Packages
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computing
- **Pillow**: Image handling
- **scikit-learn**: Metrics (F1, precision, recall)

### External Services
- **Groq Cloud**: LLM API for recommendations
- **HuggingFace**: Model weights download

---

## Data Flow Summary

```
User Upload → Validation → Preprocess → ML Inference → Post-Processing
     ↓              ↓            ↓            ↓              ↓
   UploadFile   File Type   Resize 224   EfficientNet   Grad-CAM
                Size Check  Normalize    14 Classes      Severity
                                              ↓
                                    Validation Module
                                              ↓
                                    Advisory Engine
                                              ↓
                                    Response to Client
```