"""
Decision Agent
Makes final claim decision using LLM to analyze all collected information.
"""
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from config.settings import settings
from config.logging import app_logger
from schemas.models import (
    DecisionOutput,
    DamageAnalysisOutput,
    InvoiceData,
    ClaimNLPOutput,
    PolicyValidationOutput,
    DecisionType
)
from prompts.templates import DECISION_SYSTEM_PROMPT, DECISION_USER_PROMPT
from agents.llm_utils import parse_json_response, normalize_string, coerce_float, normalize_confidence


class DecisionAgent:
    """Agent that makes final claim decisions based on all available data."""
    
    def __init__(self):
        """Initialize the LLM for decision making."""
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.0,  # Use 0 for deterministic decisions
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DECISION_SYSTEM_PROMPT),
            ("human", DECISION_USER_PROMPT)
        ])
        
        # Create chain with string output parser (we'll parse JSON ourselves)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        app_logger.info("Decision Agent initialized")
    
    def _serialize_for_llm(self, obj: Any) -> str:
        """Convert Pydantic models to readable string format for LLM."""
        if hasattr(obj, 'model_dump_json'):
            return obj.model_dump_json(indent=2)
        return str(obj)
    
    def process(
        self,
        damage_analysis: DamageAnalysisOutput,
        invoice_data: InvoiceData,
        claim_nlp: ClaimNLPOutput,
        policy_validation: PolicyValidationOutput
    ) -> DecisionOutput:
        """
        Make final claim decision based on all available information.
        
        Args:
            damage_analysis: Vehicle damage analysis
            invoice_data: Structured invoice information
            claim_nlp: Parsed claim information
            policy_validation: Policy validation results
            
        Returns:
            DecisionOutput with decision and reasoning
        """
        try:
            app_logger.info("Making claim decision")
            
            # Prepare data for LLM
            damage_str = self._serialize_for_llm(damage_analysis)
            invoice_str = self._serialize_for_llm(invoice_data)
            claim_str = self._serialize_for_llm(claim_nlp)
            policy_str = self._serialize_for_llm(policy_validation)
            
            # Invoke LLM chain
            raw_response = self.chain.invoke({
                "damage_analysis": damage_str,
                "invoice_data": invoice_str,
                "claim_info": claim_str,
                "policy_validation": policy_str
            })
            decision_data = parse_json_response(raw_response)
            normalized = self._normalize_decision_data(decision_data)
            
            # Validate and create DecisionOutput
            decision_output = DecisionOutput(**normalized)
            
            app_logger.info(f"Decision made: {decision_output.decision} with confidence {decision_output.confidence:.2f}")
            
            # Additional validation logic (override LLM if necessary)
            decision_output = self._apply_business_rules(
                decision_output,
                damage_analysis,
                invoice_data,
                claim_nlp,
                policy_validation
            )
            
            return decision_output
            
        except ValidationError as e:
            app_logger.error(f"Invalid decision data from LLM: {str(e)}")
            return self._fallback_decision("Unable to make automated decision due to invalid LLM output")
            
        except Exception as e:
            app_logger.error(f"Decision making failed: {str(e)}")
            return self._fallback_decision("Unable to make automated decision due to processing error")
    
    def _apply_business_rules(
        self,
        decision: DecisionOutput,
        damage_analysis: DamageAnalysisOutput,
        invoice_data: InvoiceData,
        claim_nlp: ClaimNLPOutput,
        policy_validation: PolicyValidationOutput
    ) -> DecisionOutput:
        """
        Apply hard business rules to override LLM decision if necessary.
        
        Args:
            decision: LLM's decision
            damage_analysis: Damage analysis results
            invoice_data: Invoice data
            policy_validation: Policy validation results
            
        Returns:
            Potentially modified DecisionOutput
        """
        
        # Rule 1: If not covered by policy, always REJECT
        if not policy_validation.covered:
            # If coverage is explicitly unknown, prefer review over reject
            if "insufficient data" in policy_validation.reason.lower() or "undetermined" in policy_validation.reason.lower():
                app_logger.info("Overriding decision to REVIEW_REQUIRED (coverage unknown)")
                decision.decision = DecisionType.REVIEW_REQUIRED
                decision.estimated_payout = 0.0
                decision.reason = policy_validation.reason
            else:
                app_logger.info("Overriding decision to REJECT (not covered)")
                decision.decision = DecisionType.REJECT
                decision.estimated_payout = 0.0
                decision.reason = policy_validation.reason
        
        # Rule 2: If severe damage, always require review
        if damage_analysis.overall_severity.value == "severe":
            if decision.decision == DecisionType.AUTO_APPROVE:
                app_logger.info("Overriding AUTO_APPROVE to REVIEW_REQUIRED (severe damage)")
                decision.decision = DecisionType.REVIEW_REQUIRED
                decision.reason += " | Severe damage requires manual review."
        
        # Rule 3: If invoice confidence is too low, require review
        if invoice_data.confidence < 0.7:
            if decision.decision == DecisionType.AUTO_APPROVE:
                app_logger.info("Overriding AUTO_APPROVE to REVIEW_REQUIRED (low OCR confidence)")
                decision.decision = DecisionType.REVIEW_REQUIRED
                decision.reason += f" | Low invoice confidence ({invoice_data.confidence:.2f}) requires verification."
        
        # Rule 4: If amount exceeds 80% of max payable, require review
        if policy_validation.covered and invoice_data.amount > (policy_validation.max_payable * 0.8):
            if decision.decision == DecisionType.AUTO_APPROVE:
                app_logger.info("Overriding AUTO_APPROVE to REVIEW_REQUIRED (high amount)")
                decision.decision = DecisionType.REVIEW_REQUIRED
                decision.reason += " | High claim amount requires manual review."
        
        # Rule 5: Ensure estimated payout is capped at max_payable
        if decision.estimated_payout > policy_validation.max_payable:
            app_logger.info(f"Capping payout from {decision.estimated_payout} to {policy_validation.max_payable}")
            decision.estimated_payout = policy_validation.max_payable
        
        # Rule 6: Missing critical information forces review
        missing_inputs: List[str] = []
        
        if invoice_data.amount <= 0 or "amount" in invoice_data.missing_fields:
            missing_inputs.append("invoice amount")
        
        if invoice_data.confidence < 0.5:
            missing_inputs.append("invoice confidence")
        
        if not claim_nlp.incident_date:
            missing_inputs.append("incident date")
        
        if not claim_nlp.location:
            missing_inputs.append("incident location")
        
        if "insufficient data" in policy_validation.reason.lower() or "undetermined" in policy_validation.reason.lower():
            missing_inputs.append("policy validation")
        
        if missing_inputs and decision.decision == DecisionType.AUTO_APPROVE:
            app_logger.info("Overriding AUTO_APPROVE to REVIEW_REQUIRED (missing data)")
            decision.decision = DecisionType.REVIEW_REQUIRED
            missing_str = ", ".join(sorted(set(missing_inputs)))
            decision.reason += f" | Missing information: {missing_str}."
        
        return decision

    def _normalize_decision_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM output into a safe DecisionOutput dict."""
        normalized = {
            "decision": "REVIEW_REQUIRED",
            "reason": "Unable to make automated decision",
            "estimated_payout": 0.0,
            "confidence": 0.0
        }
        
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    normalized[key] = value
        
        decision = normalize_string(normalized.get("decision")).upper()
        if decision not in {d.value for d in DecisionType}:
            decision = DecisionType.REVIEW_REQUIRED.value
        normalized["decision"] = decision
        
        reason = normalize_string(normalized.get("reason"))
        normalized["reason"] = reason or "Unable to make automated decision"
        
        normalized["estimated_payout"] = coerce_float(normalized.get("estimated_payout"), 0.0)
        normalized["confidence"] = normalize_confidence(normalized.get("confidence"), 0.0)
        
        return normalized

    def _fallback_decision(self, reason: str) -> DecisionOutput:
        """Create a conservative fallback decision."""
        return DecisionOutput(
            decision=DecisionType.REVIEW_REQUIRED,
            reason=reason,
            estimated_payout=0.0,
            confidence=0.0
        )


# Singleton instance
decision_agent = DecisionAgent()
