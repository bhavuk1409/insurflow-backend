# Vehicle Insurance Claims Processing AI

Production-ready multi-agent GenAI system for end-to-end vehicle insurance claims processing.

---

## System Overview

This system processes vehicle insurance claims automatically through a multi-agent pipeline:

1. **Vehicle Damage Analysis** – YOLOv8-based damage detection
2. **Invoice OCR & Structuring** – PaddleOCR + LLM for invoice data extraction
3. **Claim NLP** – LLM-based parsing of claim descriptions
4. **Policy Validation** – Rule-based validation against policy coverage
5. **Decision Making** – LLM + business rules for claim decisions
6. **Explanation Generation** – Clear explanations for stakeholders

---

## Architecture

### Tech Stack

* **Framework**: FastAPI, LangChain, LangGraph
* **LLM**: Groq API (Mixtral / LLaMA)
* **OCR**: PaddleOCR
* **Orchestration**: LangGraph state machine
* **Validation**: Pydantic

---

## Folder Structure

```
vehicle-insurance-claims-ai/
├── agents/
├── graphs/
├── schemas/
├── tools/
├── prompts/
├── config/
├── api/
├── sample_data/
├── requirements.txt
├── .env.example
└── test_api.py
```

---

## Setup Instructions

### 1. Prerequisites

* Python 3.9+
* YOLOv8 damage detection service running on port 8000
* Groq API key

### 2. Installation

```bash
cd vehicle-insurance-claims-ai

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

mkdir -p logs
```

### 3. Configuration

```bash
cp .env.example .env
# Add your GROQ_API_KEY
```

### 4. Run the Service

```bash
cd api
python main.py
```

Or:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## API Usage

### Endpoint

`POST /process-claim`

### Parameters

* `claim_description` (required)
* `vehicle_image` (optional)
* `invoice_image` (optional)
* `policy_number` (optional)

### Example Request (Python)

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
print(response.json())
```

---

## Workflow

```
START
  ↓
damage_analysis
  ↓
ocr_structuring
  ↓
claim_nlp
  ↓
policy_validation
  ↓
decision
  ↓
explanation
  ↓
END
```

---

## Key Features

* Multi-agent architecture
* Conditional workflow routing
* Business rule overrides
* Structured outputs
* Explainable AI decisions

---

## Production Considerations

### Security

* Authentication and authorization
* Input validation
* HTTPS

### Scalability

* Replace in-memory database
* Add Redis caching
* Rate limiting

### Monitoring

* Logging
* APM tools
* Performance tracking

### Data Privacy

* PII masking
* Audit logging
* Compliance (e.g., GDPR)

---

## Sample Policies

* **POL-2024-001**: Standard (collision + comprehensive + liability)
* **POL-2024-002**: Basic (collision + liability)
* **POL-2024-003**: Premium (all coverages + theft)

---

## Troubleshooting

**OCR Issues**

* Ensure PaddleOCR is installed
* Check image quality

**Damage Detection Issues**

* Verify YOLOv8 service is running
* Check configuration URL

**LLM Issues**

* Verify API key
* Check rate limits

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit a pull request

---

## Support

For issues, create a GitHub issue.
