"""
OCR tool using PaddleOCR for invoice text extraction.
Handles both images and PDFs with proper error handling.
"""
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None
from PIL import Image, ImageOps
import numpy as np
from typing import Tuple, Optional, Any
from config.logging import app_logger
from config.settings import settings


class OCRTool:
    """OCR tool for extracting text from invoices."""
    
    def __init__(self):
        """Initialize PaddleOCR with English language support."""
        self.ocr_engine = None
        if PaddleOCR is None:
            app_logger.error("PaddleOCR is not available. OCR will be disabled.")
            return
        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=settings.ocr_lang
            )
            app_logger.info("OCR engine initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize OCR engine: {str(e)}")
            self.ocr_engine = None
    
    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            app_logger.info(f"Starting OCR extraction from: {image_path}")
            
            if not self.ocr_engine:
                app_logger.error("OCR engine is not initialized; returning empty result")
                return "", 0.0
            
            # Load image
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_np = np.array(image)
            
            # Perform OCR
            result = self.ocr_engine.ocr(image_np, cls=True)
            lines = self._normalize_result(result)
            
            if not lines:
                app_logger.warning(f"No text detected in image: {image_path}")
                return "", 0.0
            
            # Extract text and confidence scores
            extracted_lines = []
            confidence_scores = []
            
            for line in lines:
                text = line[1][0]  # Extracted text
                confidence = line[1][1]  # Confidence score
                
                extracted_lines.append(text)
                confidence_scores.append(confidence)
            
            # Combine all text
            full_text = "\n".join(extracted_lines)
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            app_logger.info(f"OCR completed. Extracted {len(extracted_lines)} lines with avg confidence: {avg_confidence:.2f}")
            
            return full_text, avg_confidence
            
        except FileNotFoundError:
            app_logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            app_logger.error(f"OCR extraction failed: {str(e)}")
            raise
    
    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """
        Extract text from a PDF file by converting pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Try direct PDF text extraction first (for text-based PDFs)
            try:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                extracted_text = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        extracted_text.append(page_text)
                direct_text = "\n".join(extracted_text).strip()
                if direct_text:
                    app_logger.info("Extracted text directly from PDF (no OCR needed)")
                    return direct_text, 0.9
            except Exception:
                pass
            
            from pdf2image import convert_from_path
            
            app_logger.info(f"Converting PDF to images: {pdf_path}")
            
            if not self.ocr_engine:
                app_logger.error("OCR engine is not initialized; returning empty result")
                return "", 0.0
            
            # Convert PDF to images (higher DPI improves OCR accuracy)
            images = convert_from_path(pdf_path, dpi=300)
            
            all_text = []
            all_confidences = []
            
            for i, image in enumerate(images):
                app_logger.info(f"Processing page {i+1}/{len(images)}")
                
                # Convert PIL image to numpy array
                image = ImageOps.exif_transpose(image).convert("RGB")
                image_np = np.array(image)
                
                # Perform OCR
                result = self.ocr_engine.ocr(image_np, cls=True)
                lines = self._normalize_result(result)
                
                if lines:
                    page_text = []
                    page_confidences = []
                    
                    for line in lines:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        page_text.append(text)
                        page_confidences.append(confidence)
                    
                    all_text.extend(page_text)
                    all_confidences.extend(page_confidences)
            
            # Combine all text
            full_text = "\n".join(all_text)
            
            # Calculate average confidence
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            app_logger.info(f"PDF OCR completed. Extracted {len(all_text)} lines with avg confidence: {avg_confidence:.2f}")
            
            return full_text, avg_confidence
            
        except Exception as e:
            app_logger.error(f"PDF OCR extraction failed: {str(e)}")
            raise

    def _normalize_result(self, result: Optional[list]) -> list:
        """Normalize PaddleOCR output to a flat list of lines."""
        if not result:
            return []
        
        if isinstance(result, list):
            # Case: result is [lines]
            if len(result) == 1 and isinstance(result[0], list) and result[0]:
                if self._is_line_entry(result[0][0]):
                    return result[0]
            # Case: result is already lines
            if result and self._is_line_entry(result[0]):
                return result
        
        return []

    @staticmethod
    def _is_line_entry(obj: Any) -> bool:
        """Check if an object looks like a PaddleOCR line entry."""
        return (
            isinstance(obj, (list, tuple))
            and len(obj) == 2
            and isinstance(obj[1], (list, tuple))
            and len(obj[1]) >= 2
        )


# Singleton instance
ocr_tool = OCRTool()
