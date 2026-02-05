"""
Pydantic schemas for type validation across the system.
All data structures are strongly typed for production reliability.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


# ==================== DAMAGE DETECTION SCHEMAS ====================

class DamageType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    BROKEN_GLASS = "broken_glass"
    BUMPER_DAMAGE = "bumper_damage"
    HEADLIGHT_DAMAGE = "headlight_damage"


class Severity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DamageDetection(BaseModel):
    damage_type: DamageType
    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity


class DamageAnalysisOutput(BaseModel):
    detections: List[DamageDetection]
    overall_severity: Severity
    total_damages: int
    image_analyzed: bool = True


# ==================== OCR & INVOICE SCHEMAS ====================

class InvoiceData(BaseModel):
    invoice_number: str = Field(default="", description="Invoice number")
    garage_name: str = Field(default="", description="Garage or repair shop name")
    amount: float = Field(default=0.0, ge=0, description="Invoice amount")
    date: str = Field(default="", description="Invoice date")
    currency: str = Field(default="INR", description="Currency code")
    confidence: float = Field(ge=0.0, le=1.0, description="OCR confidence score")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing or unclear fields")


# ==================== CLAIM NLP SCHEMAS ====================

class IncidentType(str, Enum):
    COLLISION = "collision"
    THEFT = "theft"
    FIRE = "fire"
    NATURAL_DISASTER = "natural_disaster"
    VANDALISM = "vandalism"
    OTHER = "other"


class ClaimNLPOutput(BaseModel):
    incident_type: IncidentType
    incident_date: str = Field(description="Date of incident in YYYY-MM-DD format")
    location: str = Field(description="Location of incident")
    claim_summary: str = Field(description="Brief summary of the claim")


# ==================== POLICY VALIDATION SCHEMAS ====================

class PolicyCoverage(BaseModel):
    coverage_type: str
    max_limit: float
    deductible: float = 0.0


class PolicyData(BaseModel):
    policy_number: str
    coverage_types: List[str]
    max_payout: float
    exclusions: List[str]
    coverages: List[PolicyCoverage]


class PolicyValidationOutput(BaseModel):
    covered: bool
    max_payable: float
    reason: str
    applicable_coverage: Optional[str] = None


# ==================== DECISION SCHEMAS ====================

class DecisionType(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECT = "REJECT"


class DecisionOutput(BaseModel):
    decision: DecisionType
    reason: str
    estimated_payout: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


# ==================== EXPLANATION SCHEMAS ====================

class ExplanationOutput(BaseModel):
    customer_explanation: str = Field(description="User-friendly explanation for the customer")
    officer_explanation: str = Field(description="Detailed technical explanation for claims officer")
    key_factors: List[str] = Field(description="Key factors that influenced the decision")


# ==================== API REQUEST/RESPONSE SCHEMAS ====================

class ClaimRequest(BaseModel):
    claim_description: str = Field(description="Free-text description of the incident")
    policy_number: Optional[str] = Field(default=None, description="Policy number if available")


class ClaimProcessingResponse(BaseModel):
    claim_id: str = Field(description="Unique claim identifier")
    timestamp: datetime
    damage_analysis: Optional[DamageAnalysisOutput] = None
    invoice_data: Optional[InvoiceData] = None
    claim_nlp: Optional[ClaimNLPOutput] = None
    policy_validation: Optional[PolicyValidationOutput] = None
    decision: DecisionOutput
    explanation: ExplanationOutput
    raw_ocr_text: Optional[str] = Field(default=None, description="Raw OCR text (debug)")
    policy_data: Optional[PolicyData] = Field(default=None, description="Policy data used (debug)")
    processing_time_seconds: float


# ==================== LANGGRAPH STATE SCHEMA ====================

class ClaimProcessingState(BaseModel):
    """
    Central state object passed between all agents in LangGraph.
    This is the single source of truth for the entire claim processing workflow.
    """
    
    # Input data
    vehicle_image_path: Optional[str] = None
    invoice_image_path: Optional[str] = None
    claim_description: str
    policy_number: Optional[str] = None
    
    # Agent outputs
    damage_analysis: Optional[DamageAnalysisOutput] = None
    invoice_data: Optional[InvoiceData] = None
    claim_nlp: Optional[ClaimNLPOutput] = None
    policy_validation: Optional[PolicyValidationOutput] = None
    decision: Optional[DecisionOutput] = None
    explanation: Optional[ExplanationOutput] = None
    
    # Metadata
    claim_id: str
    timestamp: datetime
    processing_errors: List[str] = Field(default_factory=list)
    requires_human_review: bool = False
    
    # Intermediate data
    raw_ocr_text: Optional[str] = None
    policy_data: Optional[PolicyData] = None
    
    class Config:
        arbitrary_types_allowed = True
