"""
OCR + Structuring Agent
Combines OCR tool with LLM to extract and structure invoice data.
"""
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from config.settings import settings
from config.logging import app_logger
from tools.ocr_tool import ocr_tool
from schemas.models import InvoiceData
from prompts.templates import OCR_STRUCTURING_SYSTEM_PROMPT, OCR_STRUCTURING_USER_PROMPT
from agents.llm_utils import (
    parse_json_response,
    coerce_float,
    normalize_confidence,
    ensure_list,
    normalize_string
)


class OCRStructuringAgent:
    """Agent that performs OCR and structures invoice data using LLM."""
    
    def __init__(self):
        """Initialize the LLM for structuring."""
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", OCR_STRUCTURING_SYSTEM_PROMPT),
            ("human", OCR_STRUCTURING_USER_PROMPT)
        ])
        
        # Create chain with string output parser (we'll parse JSON ourselves)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        app_logger.info("OCR Structuring Agent initialized")
    
    def process(self, invoice_image_path: str) -> tuple[InvoiceData, str]:
        """
        Process invoice image: OCR extraction + LLM structuring.
        
        Args:
            invoice_image_path: Path to invoice image/PDF
            
        Returns:
            Tuple of (InvoiceData, raw_ocr_text)
        """
        try:
            app_logger.info(f"Processing invoice: {invoice_image_path}")
            
            # Step 1: OCR extraction
            if invoice_image_path.lower().endswith('.pdf'):
                raw_ocr_text, ocr_confidence = ocr_tool.extract_from_pdf(invoice_image_path)
            else:
                raw_ocr_text, ocr_confidence = ocr_tool.extract_text(invoice_image_path)
            
            app_logger.info(f"OCR completed with confidence: {ocr_confidence:.2f}")
            
            if not raw_ocr_text.strip():
                app_logger.warning("No text extracted from invoice")
                return InvoiceData(
                    confidence=0.0,
                    missing_fields=["invoice_number", "garage_name", "amount", "date"]
                ), ""
            
            # Step 2: LLM structuring
            app_logger.info("Structuring OCR text with LLM")

            raw_response = self.chain.invoke({"ocr_text": raw_ocr_text})
            structured_data = parse_json_response(raw_response)
            normalized = self._normalize_invoice_data(structured_data)
            
            # Validate and create InvoiceData
            invoice_data = InvoiceData(**normalized)
            
            # Adjust confidence based on OCR quality
            invoice_data.confidence = min(invoice_data.confidence, ocr_confidence)
            
            # Heuristic enrichment from raw OCR text
            invoice_data = self._enrich_from_ocr_text(invoice_data, raw_ocr_text)
            
            app_logger.info(f"Invoice structured successfully. Amount: {invoice_data.amount}, Confidence: {invoice_data.confidence:.2f}")
            
            return invoice_data, raw_ocr_text
            
        except ValidationError as e:
            app_logger.error(f"Invalid invoice data from LLM: {str(e)}")
            # Return low-confidence default
            return InvoiceData(
                confidence=0.0,
                missing_fields=["invoice_number", "garage_name", "amount", "date"]
            ), raw_ocr_text if 'raw_ocr_text' in locals() else ""
            
        except Exception as e:
            app_logger.error(f"OCR structuring failed: {str(e)}")
            return InvoiceData(
                confidence=0.0,
                missing_fields=["invoice_number", "garage_name", "amount", "date"]
            ), raw_ocr_text if 'raw_ocr_text' in locals() else ""

    def _normalize_invoice_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM output into a safe InvoiceData dict."""
        normalized = {
            "invoice_number": "",
            "garage_name": "",
            "amount": 0.0,
            "date": "",
            "currency": "INR",
            "confidence": 0.0,
            "missing_fields": []
        }
        
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    normalized[key] = value
        
        normalized["invoice_number"] = normalize_string(normalized.get("invoice_number"))
        normalized["garage_name"] = normalize_string(normalized.get("garage_name"))
        normalized["date"] = normalize_string(normalized.get("date"))
        
        currency = normalize_string(normalized.get("currency")) or "INR"
        normalized["currency"] = currency.upper()
        
        normalized["amount"] = coerce_float(normalized.get("amount"), 0.0)
        normalized["confidence"] = normalize_confidence(normalized.get("confidence"), 0.0)
        normalized["missing_fields"] = ensure_list(normalized.get("missing_fields"))
        
        return normalized

    def _enrich_from_ocr_text(self, invoice_data: InvoiceData, raw_text: str) -> InvoiceData:
        """Fill common fields from raw OCR text when LLM misses them."""
        if not raw_text:
            return invoice_data
        
        text = raw_text.strip()
        
        if not invoice_data.invoice_number:
            invoice_number = self._extract_invoice_number(text)
            if invoice_number:
                invoice_data.invoice_number = invoice_number
                self._remove_missing_field(invoice_data, "invoice_number")
        
        if invoice_data.amount <= 0:
            amount = self._extract_total_amount(text)
            if amount and amount > 0:
                invoice_data.amount = amount
                self._remove_missing_field(invoice_data, "amount")
        
        if not invoice_data.garage_name:
            garage = self._extract_garage_name(text)
            if garage:
                invoice_data.garage_name = garage
                self._remove_missing_field(invoice_data, "garage_name")
        
        if not invoice_data.date:
            date_str = self._extract_date(text)
            if date_str:
                invoice_data.date = date_str
                self._remove_missing_field(invoice_data, "date")
        
        return invoice_data

    def _extract_invoice_number(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        patterns = [
            r"(?:tax\s+invoice|invoice|inv|bill)\s*(?:no\.?|number|#|:)\s*([A-Z0-9][A-Z0-9\-/]+)",
            r"(?:inv|invoice)\s*#?\s*([A-Z0-9][A-Z0-9\-/]+)"
        ]
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            # If line contains 'invoice' and an id-like token, grab it
            if "invoice" in line.lower():
                tokens = re.findall(r"[A-Z0-9][A-Z0-9\-/]{3,}", line)
                if tokens:
                    return tokens[-1].strip()
        # Handle "Invoice No" on one line and the number on the next
        for idx, line in enumerate(lines[:-1]):
            if re.search(r"(invoice|inv|bill)\s*(no\.?|number|#|:)?$", line, re.IGNORECASE):
                candidate = lines[idx + 1]
                candidate = re.sub(r"[^A-Z0-9\-/]", "", candidate, flags=re.IGNORECASE)
                if candidate:
                    return candidate
        return ""

    def _extract_total_amount(self, text: str) -> float:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates: list[str] = []
        keywords = ("grand total", "total", "amount", "net total", "payable")
        for line in lines:
            lower = line.lower()
            if any(k in lower for k in keywords):
                nums = re.findall(r"\d[\d,]*\.?\d*", line)
                candidates.extend(nums)
        
        if not candidates:
            return 0.0
        
        values = []
        for c in candidates:
            try:
                values.append(float(c.replace(",", "")))
            except ValueError:
                continue
        
        return max(values) if values else 0.0

    def _extract_garage_name(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:5]:
            if "garage" in line.lower() or "workshop" in line.lower() or "auto" in line.lower():
                if len(line) <= 60:
                    return line
        return ""

    def _extract_date(self, text: str) -> str:
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

    @staticmethod
    def _remove_missing_field(invoice_data: InvoiceData, field: str) -> None:
        if field in invoice_data.missing_fields:
            invoice_data.missing_fields = [f for f in invoice_data.missing_fields if f != field]


# Singleton instance
ocr_structuring_agent = OCRStructuringAgent()
