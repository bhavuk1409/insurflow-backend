"""
Claim NLP Agent
Extracts structured information from free-text claim descriptions using LLM.
"""
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from config.settings import settings
from config.logging import app_logger
from schemas.models import ClaimNLPOutput, IncidentType
from prompts.templates import CLAIM_NLP_SYSTEM_PROMPT, CLAIM_NLP_USER_PROMPT
from agents.llm_utils import parse_json_response, normalize_string


class ClaimNLPAgent:
    """Agent that extracts structured information from claim descriptions."""
    
    def __init__(self):
        """Initialize the LLM for NLP processing."""
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLAIM_NLP_SYSTEM_PROMPT),
            ("human", CLAIM_NLP_USER_PROMPT)
        ])
        
        # Create chain with string output parser (we'll parse JSON ourselves)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        app_logger.info("Claim NLP Agent initialized")
    
    def process(self, claim_description: str) -> ClaimNLPOutput:
        """
        Extract structured information from claim description.
        
        Args:
            claim_description: Free-text description of the incident
            
        Returns:
            ClaimNLPOutput with structured claim information
        """
        try:
            app_logger.info("Processing claim description with NLP")
            
            # Invoke LLM chain
            raw_response = self.chain.invoke({"claim_description": claim_description})
            structured_data = parse_json_response(raw_response)
            normalized = self._normalize_claim_data(structured_data, claim_description)
            
            # Validate and create ClaimNLPOutput
            claim_nlp_output = ClaimNLPOutput(**normalized)
            
            app_logger.info(f"Claim NLP completed. Incident type: {claim_nlp_output.incident_type}")
            
            return claim_nlp_output
            
        except ValidationError as e:
            app_logger.error(f"Invalid claim data from LLM: {str(e)}")
            return self._fallback_claim(claim_description)
            
        except Exception as e:
            app_logger.error(f"Claim NLP processing failed: {str(e)}")
            return self._fallback_claim(claim_description)

    def _normalize_claim_data(self, data: Dict[str, Any], claim_description: str) -> Dict[str, Any]:
        """Normalize LLM output into a safe ClaimNLPOutput dict."""
        normalized = {
            "incident_type": "other",
            "incident_date": "",
            "location": "",
            "claim_summary": ""
        }
        
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    normalized[key] = value
        
        incident_type = normalize_string(normalized.get("incident_type")).lower()
        allowed = {it.value for it in IncidentType}
        if incident_type not in allowed:
            incident_type = "other"
        normalized["incident_type"] = incident_type
        
        incident_date = normalize_string(normalized.get("incident_date"))
        if not incident_date:
            incident_date = self._extract_date(claim_description)
        normalized["incident_date"] = incident_date
        
        normalized["location"] = normalize_string(normalized.get("location"))
        
        claim_summary = normalize_string(normalized.get("claim_summary"))
        if not claim_summary:
            claim_summary = claim_description.strip()[:200]
        normalized["claim_summary"] = claim_summary
        
        return normalized

    def _extract_date(self, text: str) -> str:
        """Extract a date-like string from free text (best-effort)."""
        if not text:
            return ""
        
        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""

    def _fallback_claim(self, claim_description: str) -> ClaimNLPOutput:
        """Fallback ClaimNLPOutput when LLM parsing fails."""
        return ClaimNLPOutput(
            incident_type="other",
            incident_date=self._extract_date(claim_description),
            location="",
            claim_summary=claim_description.strip()[:200]
        )


# Singleton instance
claim_nlp_agent = ClaimNLPAgent()
