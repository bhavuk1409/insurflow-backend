"""
FastAPI application for Vehicle Insurance Claims Processing.
Main API endpoints and application setup.
"""
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from config.settings import settings
from config.logging import app_logger
from schemas.models import ClaimProcessingResponse, DecisionOutput
from graphs.claim_workflow import claim_processing_graph, ClaimGraphState


# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Insurance Claims Processing AI",
    description="Multi-agent GenAI system for end-to-end insurance claim processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary upload directory
UPLOAD_DIR = Path("/tmp/claim_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    app_logger.info("Starting Vehicle Insurance Claims Processing API")
    app_logger.info(f"Using LLM: {settings.llm_model}")
    app_logger.info(f"Damage Detection Service: {settings.damage_detection_url}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    app_logger.info("Shutting down Vehicle Insurance Claims Processing API")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Vehicle Insurance Claims Processing AI",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "process_claim": "/process-claim (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_model": settings.llm_model
    }


async def save_upload_file(upload_file: UploadFile, claim_id: str, file_type: str) -> Optional[str]:
    """
    Save uploaded file to temporary directory.
    
    Args:
        upload_file: Uploaded file
        claim_id: Unique claim identifier
        file_type: Type of file (vehicle/invoice)
        
    Returns:
        Path to saved file or None if no file
    """
    if not upload_file:
        return None
    
    try:
        # Create unique filename
        file_extension = Path(upload_file.filename).suffix
        filename = f"{claim_id}_{file_type}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        app_logger.info(f"Saved {file_type} file: {file_path}")
        return str(file_path)
        
    except Exception as e:
        app_logger.error(f"Failed to save {file_type} file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save {file_type} file")


@app.post("/process-claim", response_model=ClaimProcessingResponse)
async def process_claim(
    claim_description: str = Form(..., description="Free-text description of the incident"),
    vehicle_image: Optional[UploadFile] = File(None, description="Vehicle damage image"),
    invoice_image: Optional[UploadFile] = File(None, description="Repair invoice image or PDF"),
    policy_number: Optional[str] = Form(None, description="Insurance policy number"),
    debug: bool = Form(False, description="Include debug fields in response")
):
    """
    Process a vehicle insurance claim end-to-end.
    
    This endpoint orchestrates the entire claim processing workflow:
    1. Vehicle damage analysis (YOLOv8)
    2. Invoice OCR and structuring (PaddleOCR + LLM)
    3. Claim description parsing (LLM)
    4. Policy validation (rule-based)
    5. Claim decision (LLM + rules)
    6. Explanation generation (LLM)
    
    Args:
        claim_description: Free-text description of the incident
        vehicle_image: Optional image of vehicle damage
        invoice_image: Optional repair invoice (image or PDF)
        policy_number: Optional policy identifier
        
    Returns:
        ClaimProcessingResponse with complete analysis and decision
    """
    start_time = time.time()
    claim_id = str(uuid.uuid4())
    
    try:
        app_logger.info(f"Processing claim {claim_id}")
        
        # Validate inputs
        if not claim_description.strip():
            raise HTTPException(status_code=400, detail="claim_description cannot be empty")
        
        if not vehicle_image and not invoice_image:
            raise HTTPException(
                status_code=400,
                detail="At least one of vehicle_image or invoice_image must be provided"
            )
        
        # Check file sizes
        max_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        if vehicle_image:
            content = await vehicle_image.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Vehicle image is empty. Please upload a valid file."
                )
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Vehicle image size exceeds {settings.max_file_size_mb}MB limit"
                )
            app_logger.info(f"Received vehicle image: {vehicle_image.filename} ({len(content)} bytes)")
            await vehicle_image.seek(0)  # Reset file pointer
        
        if invoice_image:
            content = await invoice_image.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invoice image is empty. Please upload a valid file."
                )
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invoice image size exceeds {settings.max_file_size_mb}MB limit"
                )
            app_logger.info(f"Received invoice image: {invoice_image.filename} ({len(content)} bytes)")
            await invoice_image.seek(0)  # Reset file pointer
        
        # Save uploaded files
        vehicle_path = await save_upload_file(vehicle_image, claim_id, "vehicle")
        invoice_path = await save_upload_file(invoice_image, claim_id, "invoice")
        
        # Initialize graph state
        initial_state: ClaimGraphState = {
            "vehicle_image_path": vehicle_path or "",
            "invoice_image_path": invoice_path or "",
            "claim_description": claim_description,
            "policy_number": policy_number or "",
            "damage_analysis": {},
            "invoice_data": {},
            "claim_nlp": {},
            "policy_validation": {},
            "decision": {},
            "explanation": {},
            "claim_id": claim_id,
            "processing_errors": [],
            "requires_human_review": False,
            "raw_ocr_text": "",
            "policy_data": {}
        }
        
        # Execute the graph
        app_logger.info(f"Executing claim processing graph for {claim_id}")
        final_state = await claim_processing_graph.ainvoke(initial_state)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = ClaimProcessingResponse(
            claim_id=claim_id,
            timestamp=datetime.now(),
            damage_analysis=final_state["damage_analysis"] if final_state.get("damage_analysis") else None,
            invoice_data=final_state["invoice_data"] if final_state.get("invoice_data") else None,
            claim_nlp=final_state["claim_nlp"] if final_state.get("claim_nlp") else None,
            policy_validation=final_state["policy_validation"] if final_state.get("policy_validation") else None,
            decision=DecisionOutput(**final_state["decision"]) if final_state.get("decision") else DecisionOutput(
                decision="REVIEW_REQUIRED",
                reason="Processing incomplete",
                estimated_payout=0.0,
                confidence=0.0
            ),
            explanation=final_state["explanation"] if final_state.get("explanation") else {
                "customer_explanation": "Your claim is being reviewed.",
                "officer_explanation": "Processing incomplete, manual review required.",
                "key_factors": final_state["processing_errors"]
            },
            raw_ocr_text=final_state.get("raw_ocr_text") if debug else None,
            policy_data=final_state.get("policy_data") if debug else None,
            processing_time_seconds=round(processing_time, 2)
        )
        
        app_logger.info(f"Claim {claim_id} processed successfully in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Claim processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Use 8001 since damage detection is on 8000
        reload=True,
        log_level="info"
    )
