"""
Policy Validation Agent
Rule-based validation of claims against policy coverage and limits.
"""
from typing import Optional
from config.logging import app_logger
from schemas.models import (
    PolicyValidationOutput,
    PolicyData,
    DamageAnalysisOutput,
    InvoiceData,
    ClaimNLPOutput,
    Severity
)
from tools.policy_db import get_policy, get_default_policy


class PolicyValidationAgent:
    """Agent that validates claims against policy rules."""
    
    def __init__(self):
        """Initialize the validation agent."""
        app_logger.info("Policy Validation Agent initialized")
    
    def _get_policy_data(self, policy_number: Optional[str]) -> PolicyData:
        """Get policy data from database or use default."""
        if policy_number:
            policy = get_policy(policy_number)
            if policy:
                app_logger.info(f"Policy found: {policy_number}")
                return policy
            else:
                app_logger.warning(f"Policy not found: {policy_number}, using default")
        
        return get_default_policy()
    
    def _determine_coverage_type(self, claim_nlp: ClaimNLPOutput) -> str:
        """Determine which coverage type applies based on incident type."""
        incident_to_coverage = {
            "collision": "collision",
            "theft": "theft",
            "fire": "comprehensive",
            "natural_disaster": "comprehensive",
            "vandalism": "comprehensive",
            "other": "collision"
        }
        
        return incident_to_coverage.get(claim_nlp.incident_type.value, "collision")
    
    def _check_exclusions(
        self,
        policy: PolicyData,
        claim_nlp: ClaimNLPOutput
    ) -> tuple[bool, Optional[str]]:
        """Check if claim matches any policy exclusions."""
        
        # Check claim description for exclusion keywords
        claim_text = claim_nlp.claim_summary.lower()
        
        for exclusion in policy.exclusions:
            if exclusion.replace("_", " ") in claim_text:
                return True, exclusion
        
        return False, None
    
    def process(
        self,
        damage_analysis: DamageAnalysisOutput,
        invoice_data: InvoiceData,
        claim_nlp: ClaimNLPOutput,
        policy_number: Optional[str]
    ) -> tuple[PolicyValidationOutput, PolicyData]:
        """
        Validate claim against policy rules.
        
        Args:
            damage_analysis: Output from damage detection
            invoice_data: Structured invoice information
            claim_nlp: Parsed claim information
            policy_number: Policy identifier
            
        Returns:
            Tuple of (PolicyValidationOutput, PolicyData)
        """
        try:
            app_logger.info(f"Validating claim against policy: {policy_number or 'DEFAULT'}")
            
            # Get policy data
            policy = self._get_policy_data(policy_number)
            
            # Check for exclusions
            is_excluded, exclusion_reason = self._check_exclusions(policy, claim_nlp)
            
            if is_excluded:
                app_logger.warning(f"Claim excluded due to: {exclusion_reason}")
                return PolicyValidationOutput(
                    covered=False,
                    max_payable=0.0,
                    reason=f"Claim excluded: {exclusion_reason.replace('_', ' ')}"
                ), policy
            
            # Determine applicable coverage type
            coverage_type = self._determine_coverage_type(claim_nlp)
            
            # Check if coverage type is in policy
            if coverage_type not in policy.coverage_types:
                app_logger.warning(f"Coverage type '{coverage_type}' not in policy")
                return PolicyValidationOutput(
                    covered=False,
                    max_payable=0.0,
                    reason=f"Policy does not cover {coverage_type} incidents"
                ), policy
            
            # Handle missing invoice amount explicitly
            if invoice_data.amount <= 0 and ("amount" in invoice_data.missing_fields):
                app_logger.warning("Invoice amount missing; coverage undetermined")
                return PolicyValidationOutput(
                    covered=True,
                    max_payable=0.0,
                    reason="Invoice amount missing; coverage undetermined. Manual review required.",
                    applicable_coverage=coverage_type
                ), policy
            
            # Find the specific coverage
            applicable_coverage = None
            for coverage in policy.coverages:
                if coverage.coverage_type == coverage_type:
                    applicable_coverage = coverage
                    break
            
            if not applicable_coverage:
                app_logger.error(f"Coverage configuration error for type: {coverage_type}")
                return PolicyValidationOutput(
                    covered=False,
                    max_payable=0.0,
                    reason="Coverage configuration error"
                ), policy
            
            # Calculate maximum payable amount
            # max_payable = min(invoice_amount - deductible, coverage_limit)
            invoice_amount = invoice_data.amount
            deductible = applicable_coverage.deductible
            coverage_limit = applicable_coverage.max_limit
            
            if invoice_amount <= deductible:
                app_logger.info("Invoice amount below deductible")
                return PolicyValidationOutput(
                    covered=True,
                    max_payable=0.0,
                    reason=f"Invoice amount (₹{invoice_amount}) is below deductible (₹{deductible})",
                    applicable_coverage=coverage_type
                ), policy
            
            max_payable = min(invoice_amount - deductible, coverage_limit)
            
            app_logger.info(f"Claim validated. Covered: True, Max payable: ₹{max_payable}")
            
            return PolicyValidationOutput(
                covered=True,
                max_payable=max_payable,
                reason=f"Claim covered under {coverage_type}. Max payout: ₹{max_payable} (Invoice: ₹{invoice_amount} - Deductible: ₹{deductible})",
                applicable_coverage=coverage_type
            ), policy
            
        except Exception as e:
            app_logger.error(f"Policy validation failed: {str(e)}")
            raise


# Singleton instance
policy_validation_agent = PolicyValidationAgent()
