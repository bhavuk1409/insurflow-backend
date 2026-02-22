"""
HTTP client for calling the existing YOLOv8 damage detection service.
"""
import httpx
from typing import Dict, Any
from config.logging import app_logger
from config.settings import settings
from schemas.models import DamageAnalysisOutput


class DamageDetectionClient:
    """Client for communicating with the damage detection FastAPI service."""
    
    def __init__(self):
        """Initialize the HTTP client."""
        self.base_url = settings.damage_detection_url
        self.timeout = 30.0  # seconds
    
    async def analyze_damage(self, image_path: str) -> DamageAnalysisOutput:
        """
        Call the damage detection API to analyze vehicle damage.
        Args:
            image_path: Path to the vehicle image
        Returns:
            DamageAnalysisOutput with detection results
        """
        try:
            app_logger.info(f"Calling damage detection API for: {image_path}")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(image_path, 'rb') as f:
                    files = {'file': (image_path.split('/')[-1], f, 'image/jpeg')}
                    response = await client.post(
                        f"{self.base_url}/predict",
                        files=files
                    )
                    response.raise_for_status()
                    result = response.json()
                    # --- Transform detections: bbox -> bounding_box ---
                    detections = result.get('detections', [])
                    transformed = []
                    for det in detections:
                        det = det.copy()
                        if 'bbox' in det:
                            bbox = det.pop('bbox')
                            det['bounding_box'] = {
                                'x': bbox.get('x1'),
                                'y': bbox.get('y1'),
                                'width': bbox.get('x2') - bbox.get('x1'),
                                'height': bbox.get('y2') - bbox.get('y1')
                            }
                        transformed.append(det)
                    result['detections'] = transformed
                    # --- Add overall_severity and total_damages if missing ---
                    if 'overall_severity' not in result:
                        # Use the most common severity or 'minor' as fallback
                        if transformed:
                            severities = [d.get('severity', 'minor') for d in transformed]
                            from collections import Counter
                            result['overall_severity'] = Counter(severities).most_common(1)[0][0]
                        else:
                            result['overall_severity'] = 'minor'
                    if 'total_damages' not in result:
                        result['total_damages'] = len(transformed)
                    app_logger.info(f"Damage detection completed: {result.get('total_damages', 0)} damages found")
                    # Parse into Pydantic model
                    return DamageAnalysisOutput(**result)
                    
        except httpx.HTTPError as e:
            app_logger.error(f"HTTP error calling damage detection API: {str(e)}")
            raise
        except FileNotFoundError:
            app_logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            app_logger.error(f"Damage detection failed: {str(e)}")
            raise


# Singleton instance
damage_detection_client = DamageDetectionClient()
