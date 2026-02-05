"""
Explanation Agent
Generates human-readable explanations grounded in actual data.
"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from config.settings import settings
from config.logging import app_logger
from schemas.models import (
    ExplanationOutput,
    DecisionOutput,
    DamageAnalysisOutput,
    InvoiceData,
    ClaimNLPOutput,
    PolicyValidationOutput
)
from prompts.templates import EXPLANATION_SYSTEM_PROMPT, EXPLANATION_USER_PROMPT
from agents.llm_utils import parse_json_response, normalize_string, ensure_list


class ExplanationAgent:
    """Agent that generates clear, grounded explanations for claim decisions."""
    
    def __init__(self):
        """Initialize the LLM for explanation generation."""
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.3,  # Slightly higher for natural language
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EXPLANATION_SYSTEM_PROMPT),
            ("human", EXPLANATION_USER_PROMPT)
        ])
        
        # Create chain with string output parser (we'll parse JSON ourselves)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        app_logger.info("Explanation Agent initialized")
    
    def _serialize_for_llm(self, obj: Any) -> str:
        """Convert Pydantic models to readable string format for LLM."""
        if hasattr(obj, 'model_dump_json'):
            return obj.model_dump_json(indent=2)
        return str(obj)
    
    def process(
        self,
        decision: DecisionOutput,
        damage_analysis: DamageAnalysisOutput,
        invoice_data: InvoiceData,
        claim_nlp: ClaimNLPOutput,
        policy_validation: PolicyValidationOutput
    ) -> ExplanationOutput:
        """
        Generate explanations for the claim decision.
        
        Args:
            decision: Final decision made
            damage_analysis: Vehicle damage analysis
            invoice_data: Structured invoice information
            claim_nlp: Parsed claim information
            policy_validation: Policy validation results
            
        Returns:
            ExplanationOutput with customer and officer explanations
        """
        try:
            app_logger.info("Generating claim explanations")
            
            # Prepare data for LLM
            decision_str = self._serialize_for_llm(decision)
            damage_str = self._serialize_for_llm(damage_analysis)
            invoice_str = self._serialize_for_llm(invoice_data)
            claim_str = self._serialize_for_llm(claim_nlp)
            policy_str = self._serialize_for_llm(policy_validation)
            
            # Invoke LLM chain
            raw_response = self.chain.invoke({
                "decision": decision_str,
                "damage_analysis": damage_str,
                "invoice_data": invoice_str,
                "claim_info": claim_str,
                "policy_validation": policy_str
            })
            explanation_data = parse_json_response(raw_response)
            normalized = self._normalize_explanation_data(explanation_data, decision, damage_analysis, policy_validation)
            
            # Validate and create ExplanationOutput
            explanation_output = ExplanationOutput(**normalized)
            
            app_logger.info("Explanations generated successfully")
            
            return explanation_output
            
        except ValidationError as e:
            app_logger.error(f"Invalid explanation data from LLM: {str(e)}")
            return self._fallback_explanation(decision, damage_analysis, policy_validation)
            
        except Exception as e:
            app_logger.error(f"Explanation generation failed: {str(e)}")
            return self._fallback_explanation(decision, damage_analysis, policy_validation)

    def _normalize_explanation_data(
        self,
        data: Dict[str, Any],
        decision: DecisionOutput,
        damage_analysis: DamageAnalysisOutput,
        policy_validation: PolicyValidationOutput
    ) -> Dict[str, Any]:
        """Normalize LLM output into a safe ExplanationOutput dict."""
        normalized = {
            "customer_explanation": "",
            "officer_explanation": "",
            "key_factors": []
        }
        
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    normalized[key] = value
        
        normalized["customer_explanation"] = normalize_string(normalized.get("customer_explanation"))
        normalized["officer_explanation"] = normalize_string(normalized.get("officer_explanation"))
        normalized["key_factors"] = ensure_list(normalized.get("key_factors"))
        
        if not normalized["customer_explanation"] or not normalized["officer_explanation"]:
            fallback = self._fallback_explanation(decision, damage_analysis, policy_validation)
            normalized["customer_explanation"] = normalized["customer_explanation"] or fallback.customer_explanation
            normalized["officer_explanation"] = normalized["officer_explanation"] or fallback.officer_explanation
        
        if not normalized["key_factors"]:
            normalized["key_factors"] = [
                f"Decision: {decision.decision.value}",
                f"Covered: {policy_validation.covered}",
                f"Damage severity: {damage_analysis.overall_severity.value}"
            ]
        
        return normalized

    def _fallback_explanation(
        self,
        decision: DecisionOutput,
        damage_analysis: DamageAnalysisOutput,
        policy_validation: PolicyValidationOutput
    ) -> ExplanationOutput:
        """Fallback explanation when LLM parsing fails."""
        return ExplanationOutput(
            customer_explanation=f"Your claim has been {decision.decision.value}. {decision.reason}",
            officer_explanation=f"Decision: {decision.decision.value}. Reason: {decision.reason}. Estimated payout: ₹{decision.estimated_payout}",
            key_factors=[
                f"Decision: {decision.decision.value}",
                f"Covered: {policy_validation.covered}",
                f"Damage severity: {damage_analysis.overall_severity.value}"
            ]
        )


# Singleton instance
explanation_agent = ExplanationAgent()
