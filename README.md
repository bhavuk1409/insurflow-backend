# Vehicle Insurance Claims Processing AI

A multi-agent **LangGraph** pipeline that takes a raw claim (photo + invoice +
free-text description) and turns it into a structured, explainable decision —
approve, deny, or flag for human review — end to end, with no manual triage
step in between.

![python](https://img.shields.io/badge/python-3.12-3776ab) ![backend](https://img.shields.io/badge/backend-FastAPI-009688) ![orchestration](https://img.shields.io/badge/orchestration-LangGraph-1C3C3C) ![llm](https://img.shields.io/badge/LLM-Groq%20(Mixtral%2FLLaMA)-F55036) ![ocr](https://img.shields.io/badge/OCR-PaddleOCR-00AEEF) ![deploy](https://img.shields.io/badge/deploy-Docker%20%2B%20EC2-2496ED)

---

## What it does

`POST /process-claim` runs a claim through six agents in sequence, each
handing structured state to the next:

| Step | Agent | What it does |
| --- | --- | --- |
| 1 | **Damage analysis** | Calls out to a separate YOLOv8 vehicle-damage-detection service (`car_detection_api`) over HTTP and normalizes its response into detections + overall severity |
| 2 | **OCR & structuring** | PaddleOCR extracts text from the repair invoice (image or PDF), then an LLM structures it into invoice number, garage, amount, date |
| 3 | **Claim NLP** | LLM parses the free-text claim description into incident type, date, location, and a summary |
| 4 | **Policy validation** | Rule-based check of the claim against the policy's coverage (collision / comprehensive / liability / theft) and payout limits |
| 5 | **Decision** | LLM + business rules combine damage severity, invoice amount, and policy validation into `APPROVE` / `DENY` / `REVIEW_REQUIRED` with an estimated payout and confidence score |
| 6 | **Explanation** | Generates a plain-language explanation for the customer and a more technical one for the claims officer |

Any step can fail independently — the graph degrades to safe defaults (e.g.
`REVIEW_REQUIRED`, empty invoice fields) rather than crashing the whole
request, and failures are collected in `processing_errors`.

```
START → damage_analysis → ocr_structuring → claim_nlp
      → policy_validation → decision → explanation → END
```

---

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service info + endpoint list |
| GET | `/health` | Health probe (includes configured LLM model) |
| POST | `/process-claim` | Run a claim through the full agent pipeline |

**`POST /process-claim`** — `multipart/form-data`:

| Field | Required | Notes |
| --- | --- | --- |
| `claim_description` | ✅ | Free-text description of the incident |
| `vehicle_image` | at least one of `vehicle_image` / `invoice_image` | Photo of vehicle damage |
| `invoice_image` | at least one of the two | Repair invoice, image or PDF |
| `policy_number` | optional | Looked up against the (in-memory) policy database |
| `debug` | optional | If true, includes raw OCR text and policy data in the response |

```bash
curl -X POST http://localhost:8001/process-claim \
  -F "claim_description=Collision on highway near Mumbai. Front bumper damaged." \
  -F "policy_number=POL-2024-001" \
  -F "vehicle_image=@damaged_car.jpg" \
  -F "invoice_image=@repair_invoice.pdf"
```

Response includes `damage_analysis`, `invoice_data`, `claim_nlp`,
`policy_validation`, `decision`, `explanation`, and `processing_time_seconds`.

---

## Repository layout

```
├── api/
│   └── main.py                  # FastAPI app, /process-claim endpoint, file upload handling
├── graphs/
│   └── claim_workflow.py        # LangGraph state machine wiring the 6 agents together
├── agents/
│   ├── ocr_structuring_agent.py
│   ├── claim_nlp_agent.py
│   ├── policy_validation_agent.py
│   ├── decision_agent.py
│   ├── explanation_agent.py
│   └── llm_utils.py
├── tools/
│   ├── damage_detection_client.py   # HTTP client for the external YOLOv8 damage-detection API
│   ├── ocr_tool.py                   # PaddleOCR wrapper
│   └── policy_db.py                   # In-memory sample policy database
├── schemas/models.py                  # Pydantic models (request/response, per-agent outputs)
├── prompts/templates.py                # LLM prompt templates
├── config/settings.py, logging.py       # Env-driven settings, loguru logging
├── sample_data/invoice.pdf               # Sample invoice for testing
├── .github/workflows/deploy.yml          # Build → push to Docker Hub → deploy to EC2 on push to main
├── Dockerfile
└── requirements.txt
```

---

## Setup

**Prerequisites**
- Python 3.9+ (Docker image uses 3.12)
- The [`car_detection_api`](https://github.com/bhavuk1409/car_detection_api) damage-detection service running (default `http://localhost:8000`)
- A Groq API key

**Install**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p logs
```

**Configure** — create a `.env` with at least:

```bash
GROQ_API_KEY=your_key_here
DAMAGE_DETECTION_URL=http://localhost:8000   # defaults to this if unset
LLM_MODEL=mixtral-8x7b-32768                  # default
MAX_FILE_SIZE_MB=10                            # default
```

**Run**

```bash
cd api && python main.py
# or
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

Runs on port `8001` (the damage-detection service occupies `8000`).

---

## Sample policies

The bundled in-memory policy database (`tools/policy_db.py`) ships three
test policies:

| Policy number | Tier | Coverage |
| --- | --- | --- |
| `POL-2024-001` | Standard | Collision + comprehensive + liability |
| `POL-2024-002` | Basic | Collision + liability |
| `POL-2024-003` | Premium | All coverages + theft |

---

## Deployment

`deploy.yml` runs on every push to `main`: builds the Docker image, pushes it
to Docker Hub, then SSHes into an EC2 instance to prune old images, pull the
new one, and restart the `fastapi-app` container on port `8001` with
`--restart unless-stopped`. Required GitHub secrets: `DOCKERHUB_USERNAME`,
`DOCKERHUB_TOKEN`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`.

```bash
docker build -t insurflow-backend .
docker run -d -p 8001:8001 --name fastapi-app insurflow-backend
```

---

## Production considerations

The current setup is demo-grade in a few specific ways worth calling out
before going live:

- **Policy data is in-memory** (`tools/policy_db.py`) — replace with a real
  database before production use.
- **CORS is wide open** (`allow_origins=["*"]`) — restrict this.
- **No auth** on `/process-claim` — add authentication/authorization.
- Add rate limiting, PII masking, and audit logging for compliance.

---

## Tech stack

- **FastAPI** — HTTP API
- **LangChain + LangGraph** — agent orchestration / state machine
- **Groq API** (Mixtral / LLaMA) — LLM inference
- **PaddleOCR** (+ pytesseract, pdf2image fallback path) — invoice OCR
- **Pydantic** — structured validation throughout the pipeline
- **loguru** — logging

---

## License

MIT
