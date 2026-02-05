# Vehicle Insurance Claims Processing AI

Production-ready multi-agent GenAI system for end-to-end vehicle insurance claims processing.

## 🎯 System Overview

This system processes vehicle insurance claims automatically through a sophisticated multi-agent pipeline:

1. **Vehicle Damage Analysis** - YOLOv8-based damage detection (existing service)
2. **Invoice OCR & Structuring** - PaddleOCR + LLM for invoice data extraction
3. **Claim NLP** - LLM-based parsing of claim descriptions
4. **Policy Validation** - Rule-based validation against policy coverage
5. **Decision Making** - LLM + business rules for claim decisions
6. **Explanation Generation** - Clear, grounded explanations for stakeholders

## 🏗️ Architecture

### Tech Stack
- **Framework**: FastAPI + LangChain + LangGraph
- **LLM**: Groq API (Mixtral/LLaMA)
- **OCR**: PaddleOCR
- **Orchestration**: LangGraph state machine
- **Validation**: Pydantic schemas

### Folder Structure
```
vehicle-insurance-claims-ai/
├── agents/                          # Individual AI agents
│   ├── ocr_structuring_agent.py    # OCR + LLM structuring
│   ├── claim_nlp_agent.py          # Claim description parser
│   ├── policy_validation_agent.py  # Policy rule validator
│   ├── decision_agent.py           # Decision maker
│   └── explanation_agent.py        # Explanation generator
├── graphs/                          # LangGraph orchestration
│   └── claim_workflow.py           # Main workflow graph
├── schemas/                         # Pydantic models
│   └── models.py                   # All data structures
├── tools/                           # External tools & clients
│   ├── ocr_tool.py                 # PaddleOCR wrapper
│   ├── damage_detection_client.py  # YOLOv8 service client
│   └── policy_db.py                # Sample policy database
├── prompts/                         # LLM prompt templates
│   └── templates.py                # All agent prompts
├── config/                          # Configuration
│   ├── settings.py                 # Environment settings
│   └── logging.py                  # Loguru setup
├── api/                             # FastAPI application
│   └── main.py                     # Main API endpoints
├── sample_data/                     # Sample test data
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── test_api.py                     # Test client
```

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.9+
- YOLOv8 damage detection service running on port 8000
- Groq API key ([get one here](https://console.groq.com))

### 2. Installation

```bash
# Navigate to project directory
cd /Users/bhavukagrawal/Desktop/vehicle-insurance-claims-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create logs directory
mkdir -p logs
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_actual_api_key_here
```

### 4. Run the Service

```bash
# Start the API server
cd api
python main.py

# Or use uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://localhost:8001`

## 📡 API Usage

### Endpoint: POST `/process-claim`

**Parameters:**
- `claim_description` (required): Free-text description of the incident
- `vehicle_image` (optional): Image of vehicle damage
- `invoice_image` (optional): Repair invoice image or PDF
- `policy_number` (optional): Insurance policy number

**Example Request (Python):**

```python
import requests

url = "http://localhost:8001/process-claim"

data = {
    "claim_description": "Collision on highway near Mumbai. Front bumper damaged.",
    "policy_number": "POL-2024-001"
}

files = {
    "vehicle_image": open("damaged_car.jpg", "rb"),
    "invoice_image": open("repair_invoice.pdf", "rb")
}

response = requests.post(url, data=data, files=files)
result = response.json()

print(f"Decision: {result['decision']['decision']}")
print(f"Payout: ₹{result['decision']['estimated_payout']}")
```

**Example Request (curl):**

```bash
curl -X POST http://localhost:8001/process-claim \
  -F "claim_description=Front bumper damage from parking lot collision" \
  -F "vehicle_image=@damaged_car.jpg" \
  -F "invoice_image=@invoice.pdf" \
  -F "policy_number=POL-2024-001"
```

### Response Structure

```json
{
  "claim_id": "uuid",
  "timestamp": "2026-02-05T10:30:00",
  "decision": {
    "decision": "AUTO_APPROVE",
    "reason": "Claim covered under collision coverage",
    "estimated_payout": 25000,
    "confidence": 0.95
  },
  "damage_analysis": {
    "total_damages": 2,
    "overall_severity": "moderate",
    "detections": [...]
  },
  "invoice_data": {
    "garage_name": "ABC Auto Repair",
    "amount": 30000,
    "confidence": 0.92
  },
  "policy_validation": {
    "covered": true,
    "max_payable": 25000
  },
  "explanation": {
    "customer_explanation": "Your claim has been approved...",
    "officer_explanation": "Technical details...",
    "key_factors": [...]
  },
  "processing_time_seconds": 5.2
}
```

## 🧪 Testing

Run the test client:

```bash
# Test with full data (requires sample images)
python test_api.py

# Test with minimal data (description only)
python test_api.py minimal
```

## 🔄 LangGraph Workflow

The system uses a state machine with the following flow:

```
START
  ↓
damage_analysis (async call to YOLOv8 service)
  ↓
ocr_structuring (PaddleOCR + LLM)
  ↓
claim_nlp (LLM parsing)
  ↓
[conditional: has required data?]
  ↓
policy_validation (rule-based)
  ↓
decision (LLM + business rules)
  ↓
explanation (LLM)
  ↓
END
```

**Key Features:**
- State passed between all agents
- Conditional routing based on data availability
- Error handling at each node
- Business rules override LLM when necessary

## 🎛️ Agent Details

### 1. OCR Structuring Agent
- **Input**: Invoice image/PDF
- **Process**: PaddleOCR → LLM structuring
- **Output**: Structured JSON with invoice data
- **Handles**: Noisy OCR, missing fields, confidence scoring

### 2. Claim NLP Agent
- **Input**: Free-text claim description
- **Process**: LLM extraction with strict schema
- **Output**: Incident type, date, location, summary
- **Key**: Zero hallucination prompting

### 3. Policy Validation Agent
- **Input**: Damage + invoice + claim data
- **Process**: Rule-based validation
- **Output**: Coverage decision, max payout
- **Features**: Exclusion checking, coverage mapping

### 4. Decision Agent
- **Input**: All previous outputs
- **Process**: LLM analysis + business rule overrides
- **Output**: AUTO_APPROVE / REVIEW_REQUIRED / REJECT
- **Safety**: Conservative routing, human review triggers

### 5. Explanation Agent
- **Input**: Decision + all supporting data
- **Process**: LLM with grounding constraints
- **Output**: Customer + officer explanations
- **Key**: Grounded in actual data, no hallucinations

## ⚙️ Configuration

Key settings in `.env`:

```bash
# LLM Configuration
GROQ_API_KEY=your_key
LLM_MODEL=mixtral-8x7b-32768  # or llama-3.1-70b-versatile
LLM_TEMPERATURE=0.1

# Damage Detection Service
DAMAGE_DETECTION_URL=http://localhost:8000

# OCR
OCR_ENGINE=paddleocr
OCR_LANG=en

# Policy Limits
DEFAULT_MAX_PAYOUT=500000
DEFAULT_CURRENCY=INR
```

## 🔒 Production Considerations

1. **Security**
   - Add authentication/authorization
   - Validate file types and sizes
   - Sanitize inputs
   - Use HTTPS in production

2. **Scalability**
   - Replace in-memory policy DB with actual database
   - Add Redis for caching
   - Implement rate limiting
   - Use async everywhere

3. **Monitoring**
   - Logs are in `logs/` directory
   - Add APM (e.g., Datadog, New Relic)
   - Track agent performance metrics
   - Monitor LLM costs

4. **Data Privacy**
   - Implement PII detection/masking
   - Add audit logging
   - Ensure GDPR compliance
   - Secure file storage

## 📊 Sample Policies

Three sample policies are included in `tools/policy_db.py`:

- **POL-2024-001**: Standard (collision + comprehensive + liability)
- **POL-2024-002**: Basic (collision + liability)
- **POL-2024-003**: Premium (all coverages + theft)

Replace with actual database queries in production.

## 🐛 Troubleshooting

**Issue**: OCR fails
- Ensure PaddleOCR is installed: `pip install paddleocr paddlepaddle`
- Check image quality and format

**Issue**: Damage detection fails
- Verify YOLOv8 service is running on port 8000
- Check DAMAGE_DETECTION_URL in .env

**Issue**: LLM errors
- Verify Groq API key is valid
- Check API rate limits
- Review prompt templates

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📧 Support

For issues and questions, please create a GitHub issue.

---

**Built with ❤️ for production-grade AI claims processing**
