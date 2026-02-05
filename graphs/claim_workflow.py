"""
LangGraph orchestration for the claim processing workflow.
Defines the agent graph with conditional routing and state management.
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config.logging import app_logger
from schemas.models import (
    ClaimProcessingState,
    DamageAnalysisOutput,
    InvoiceData,
    ClaimNLPOutput,
    PolicyValidationOutput,
    DecisionOutput,
    ExplanationOutput,
    DecisionType,
    Severity
)
from agents.ocr_structuring_agent import ocr_structuring_agent
from agents.claim_nlp_agent import claim_nlp_agent
from agents.policy_validation_agent import policy_validation_agent
from agents.decision_agent import decision_agent
from agents.explanation_agent import explanation_agent
from tools.damage_detection_client import damage_detection_client
from tools.policy_db import get_default_policy


class ClaimGraphState(TypedDict):
    """State dictionary for LangGraph (must be TypedDict for LangGraph compatibility)."""
    # Input data
    vehicle_image_path: str
    invoice_image_path: str
    claim_description: str
    policy_number: str
    
    # Agent outputs
    damage_analysis: dict
    invoice_data: dict
    claim_nlp: dict
    policy_validation: dict
    decision: dict
    explanation: dict
    
    # Metadata
    claim_id: str
    processing_errors: list
    requires_human_review: bool
    raw_ocr_text: str
    policy_data: dict


def _default_damage_analysis(image_analyzed: bool = False) -> DamageAnalysisOutput:
    """Create a safe default damage analysis output."""
    return DamageAnalysisOutput(
        detections=[],
        overall_severity=Severity.MINOR,
        total_damages=0,
        image_analyzed=image_analyzed
    )


def _default_invoice_data() -> InvoiceData:
    """Create a safe default invoice data output."""
    return InvoiceData(
        confidence=0.0,
        missing_fields=["invoice_number", "garage_name", "amount", "date"]
    )


def _default_claim_nlp(description: str) -> ClaimNLPOutput:
    """Create a safe default claim NLP output."""
    summary = (description or "").strip()[:200]
    return ClaimNLPOutput(
        incident_type="other",
        incident_date="",
        location="",
        claim_summary=summary
    )


def _default_policy_validation(reason: str) -> PolicyValidationOutput:
    """Create a safe default policy validation output."""
    return PolicyValidationOutput(
        covered=True,
        max_payable=0.0,
        reason=reason,
        applicable_coverage=None
    )


def _default_decision(reason: str) -> DecisionOutput:
    """Create a safe default decision output."""
    return DecisionOutput(
        decision=DecisionType.REVIEW_REQUIRED,
        reason=reason,
        estimated_payout=0.0,
        confidence=0.0
    )


def _default_explanation(errors: list) -> ExplanationOutput:
    """Create a safe default explanation output."""
    factors = errors if errors else ["Processing incomplete"]
    return ExplanationOutput(
        customer_explanation="Your claim is being reviewed.",
        officer_explanation="Processing incomplete, manual review required.",
        key_factors=[str(f) for f in factors]
    )


# ==================== NODE FUNCTIONS ====================

async def damage_analysis_node(state: ClaimGraphState) -> ClaimGraphState:
    """
    Node 1: Analyze vehicle damage using existing YOLOv8 service.
    """
    try:
        app_logger.info("Executing damage_analysis_node")
        
        if not state.get("vehicle_image_path"):
            app_logger.warning("No vehicle image provided, skipping damage analysis")
            state["processing_errors"].append("No vehicle image provided")
            state["damage_analysis"] = _default_damage_analysis(image_analyzed=False).model_dump()
            return state
        
        # Call damage detection service
        damage_result = await damage_detection_client.analyze_damage(state["vehicle_image_path"])
        
        # Store result in state
        state["damage_analysis"] = damage_result.model_dump()
        
        app_logger.info(f"Damage analysis complete: {damage_result.total_damages} damages found")
        
    except Exception as e:
        app_logger.error(f"Damage analysis node failed: {str(e)}")
        state["processing_errors"].append(f"Damage analysis failed: {str(e)}")
        state["damage_analysis"] = _default_damage_analysis(image_analyzed=False).model_dump()
    
    return state


def ocr_structuring_node(state: ClaimGraphState) -> ClaimGraphState:
    """
    Node 2: Extract and structure invoice data using OCR + LLM.
    """
    try:
        app_logger.info("Executing ocr_structuring_node")
        
        if not state.get("invoice_image_path"):
            app_logger.warning("No invoice image provided, skipping OCR")
            state["processing_errors"].append("No invoice image provided")
            state["invoice_data"] = _default_invoice_data().model_dump()
            state["raw_ocr_text"] = ""
            return state
        
        # Process invoice
        invoice_result, raw_ocr = ocr_structuring_agent.process(state["invoice_image_path"])
        
        # Store results in state
        state["invoice_data"] = invoice_result.model_dump()
        state["raw_ocr_text"] = raw_ocr
        
        app_logger.info(f"Invoice structured: Amount ₹{invoice_result.amount}, Confidence {invoice_result.confidence:.2f}")
        
    except Exception as e:
        app_logger.error(f"OCR structuring node failed: {str(e)}")
        state["processing_errors"].append(f"Invoice processing failed: {str(e)}")
        state["invoice_data"] = _default_invoice_data().model_dump()
        state["raw_ocr_text"] = ""
    
    return state


def claim_nlp_node(state: ClaimGraphState) -> ClaimGraphState:
    """
    Node 3: Parse claim description using NLP agent.
    """
    try:
        app_logger.info("Executing claim_nlp_node")
        
        if not state.get("claim_description"):
            app_logger.warning("No claim description provided, using defaults")
            claim_result = _default_claim_nlp("")
        else:
            # Process claim description
            claim_result = claim_nlp_agent.process(state["claim_description"])
        
        # Store result in state
        state["claim_nlp"] = claim_result.model_dump()
        
        app_logger.info(f"Claim parsed: Type {claim_result.incident_type}")
        
    except Exception as e:
        app_logger.error(f"Claim NLP node failed: {str(e)}")
        state["processing_errors"].append(f"Claim parsing failed: {str(e)}")
        claim_result = _default_claim_nlp(state.get("claim_description", ""))
        state["claim_nlp"] = claim_result.model_dump()
    
    return state


def policy_validation_node(state: ClaimGraphState) -> ClaimGraphState:
    """
    Node 4: Validate claim against policy rules.
    """
    try:
        app_logger.info("Executing policy_validation_node")
        
        # Reconstruct Pydantic models from state with defaults
        try:
            damage_analysis = DamageAnalysisOutput(**state["damage_analysis"]) if state.get("damage_analysis") else _default_damage_analysis()
        except Exception:
            damage_analysis = _default_damage_analysis()
            state["damage_analysis"] = damage_analysis.model_dump()
        
        try:
            invoice_data = InvoiceData(**state["invoice_data"]) if state.get("invoice_data") else _default_invoice_data()
        except Exception:
            invoice_data = _default_invoice_data()
            state["invoice_data"] = invoice_data.model_dump()
        
        try:
            claim_nlp = ClaimNLPOutput(**state["claim_nlp"]) if state.get("claim_nlp") else _default_claim_nlp(state.get("claim_description", ""))
        except Exception:
            claim_nlp = _default_claim_nlp(state.get("claim_description", ""))
            state["claim_nlp"] = claim_nlp.model_dump()
        
        # Validate against policy
        validation_result, policy_data = policy_validation_agent.process(
            damage_analysis,
            invoice_data,
            claim_nlp,
            state.get("policy_number")
        )
        
        # Store results in state
        state["policy_validation"] = validation_result.model_dump()
        state["policy_data"] = policy_data.model_dump()
        
        app_logger.info(f"Policy validation complete: Covered={validation_result.covered}")
        
    except Exception as e:
        app_logger.error(f"Policy validation node failed: {str(e)}")
        state["processing_errors"].append(f"Policy validation failed: {str(e)}")
        fallback = _default_policy_validation("Policy validation failed; manual review required.")
        state["policy_validation"] = fallback.model_dump()
        state["policy_data"] = get_default_policy().model_dump()
    
    return state


def decision_node(state: ClaimGraphState) -> ClaimGraphState:
    """
    Node 5: Make final claim decision.
    """
    try:
        app_logger.info("Executing decision_node")
        
        # Reconstruct Pydantic models from state with defaults
        try:
            damage_analysis = DamageAnalysisOutput(**state["damage_analysis"]) if state.get("damage_analysis") else _default_damage_analysis()
        except Exception:
            damage_analysis = _default_damage_analysis()
            state["damage_analysis"] = damage_analysis.model_dump()
        
        try:
            invoice_data = InvoiceData(**state["invoice_data"]) if state.get("invoice_data") else _default_invoice_data()
        except Exception:
            invoice_data = _default_invoice_data()
            state["invoice_data"] = invoice_data.model_dump()
        
        try:
            claim_nlp = ClaimNLPOutput(**state["claim_nlp"]) if state.get("claim_nlp") else _default_claim_nlp(state.get("claim_description", ""))
        except Exception:
            claim_nlp = _default_claim_nlp(state.get("claim_description", ""))
            state["claim_nlp"] = claim_nlp.model_dump()
        
        try:
            policy_validation = PolicyValidationOutput(**state["policy_validation"]) if state.get("policy_validation") else _default_policy_validation("Policy validation missing; manual review required.")
        except Exception:
            policy_validation = _default_policy_validation("Policy validation missing; manual review required.")
            state["policy_validation"] = policy_validation.model_dump()
        
        # Make decision
        decision_result = decision_agent.process(
            damage_analysis,
            invoice_data,
            claim_nlp,
            policy_validation
        )
        
        # Store result in state
        state["decision"] = decision_result.model_dump()
        
        # Set review flag
        if decision_result.decision == DecisionType.REVIEW_REQUIRED:
            state["requires_human_review"] = True
        
        app_logger.info(f"Decision made: {decision_result.decision}")
        
    except Exception as e:
        app_logger.error(f"Decision node failed: {str(e)}")
        state["processing_errors"].append(f"Decision making failed: {str(e)}")
        # Default to review required on error
        state["requires_human_review"] = True
        decision_result = _default_decision("Decision processing failed; manual review required.")
        state["decision"] = decision_result.model_dump()
    
    return state


def explanation_node(state: ClaimGraphState) -> ClaimGraphState:
    """
    Node 6: Generate explanations for the decision.
    """
    try:
        app_logger.info("Executing explanation_node")
        
        # Reconstruct Pydantic models from state with defaults
        try:
            decision = DecisionOutput(**state["decision"]) if state.get("decision") else _default_decision("Decision missing; manual review required.")
        except Exception:
            decision = _default_decision("Decision missing; manual review required.")
            state["decision"] = decision.model_dump()
        
        try:
            damage_analysis = DamageAnalysisOutput(**state["damage_analysis"]) if state.get("damage_analysis") else _default_damage_analysis()
        except Exception:
            damage_analysis = _default_damage_analysis()
            state["damage_analysis"] = damage_analysis.model_dump()
        
        try:
            invoice_data = InvoiceData(**state["invoice_data"]) if state.get("invoice_data") else _default_invoice_data()
        except Exception:
            invoice_data = _default_invoice_data()
            state["invoice_data"] = invoice_data.model_dump()
        
        try:
            claim_nlp = ClaimNLPOutput(**state["claim_nlp"]) if state.get("claim_nlp") else _default_claim_nlp(state.get("claim_description", ""))
        except Exception:
            claim_nlp = _default_claim_nlp(state.get("claim_description", ""))
            state["claim_nlp"] = claim_nlp.model_dump()
        
        try:
            policy_validation = PolicyValidationOutput(**state["policy_validation"]) if state.get("policy_validation") else _default_policy_validation("Policy validation missing; manual review required.")
        except Exception:
            policy_validation = _default_policy_validation("Policy validation missing; manual review required.")
            state["policy_validation"] = policy_validation.model_dump()
        
        # Generate explanations
        explanation_result = explanation_agent.process(
            decision,
            damage_analysis,
            invoice_data,
            claim_nlp,
            policy_validation
        )
        
        # Store result in state
        state["explanation"] = explanation_result.model_dump()
        
        app_logger.info("Explanations generated")
        
    except Exception as e:
        app_logger.error(f"Explanation node failed: {str(e)}")
        state["processing_errors"].append(f"Explanation generation failed: {str(e)}")
        fallback = _default_explanation(state.get("processing_errors", []))
        state["explanation"] = fallback.model_dump()
    
    return state


# ==================== CONDITIONAL ROUTING ====================

def should_continue_after_data_collection(state: ClaimGraphState) -> Literal["policy_validation", "end"]:
    """
    Routing function after data collection phase.
    If critical data is missing, end early with error.
    """
    # Check if we have minimum required data
    has_damage = bool(state.get("damage_analysis"))
    has_invoice = bool(state.get("invoice_data"))
    has_claim = bool(state.get("claim_nlp"))
    
    if not (has_damage or has_invoice or has_claim):
        app_logger.error("No data collected from any source, cannot proceed")
        return "end"
    
    # If we have at least some data, proceed
    return "policy_validation"


# ==================== GRAPH CONSTRUCTION ====================

def create_claim_processing_graph():
    """
    Create and compile the claim processing workflow graph.
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Initialize the graph
    workflow = StateGraph(ClaimGraphState)
    
    # Add nodes
    workflow.add_node("damage_analysis", damage_analysis_node)
    workflow.add_node("ocr_structuring", ocr_structuring_node)
    workflow.add_node("claim_nlp", claim_nlp_node)
    workflow.add_node("policy_validation", policy_validation_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("explanation", explanation_node)
    
    # Define edges (workflow)
    # Phase 1: Parallel data collection
    workflow.set_entry_point("damage_analysis")
    workflow.add_edge("damage_analysis", "ocr_structuring")
    workflow.add_edge("ocr_structuring", "claim_nlp")
    
    # Phase 2: Sequential processing
    workflow.add_conditional_edges(
        "claim_nlp",
        should_continue_after_data_collection,
        {
            "policy_validation": "policy_validation",
            "end": END
        }
    )
    
    workflow.add_edge("policy_validation", "decision")
    workflow.add_edge("decision", "explanation")
    workflow.add_edge("explanation", END)
    
    # Compile the graph
    app = workflow.compile()
    
    app_logger.info("Claim processing graph compiled successfully")
    
    return app


# Create the compiled graph
claim_processing_graph = create_claim_processing_graph()
