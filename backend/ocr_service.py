"""
OCR Service for Patient Number Extraction
Local OCR using Tesseract (no API calls for privacy)
"""

import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
import base64

class PatientNumberOCR:
    """
    Local OCR service for extracting patient numbers from photos
    Uses Tesseract OCR for privacy (no external API calls)
    """
    
    def __init__(self):
        # Configure Tesseract for better number recognition
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        print("🔍 Patient Number OCR initialized with Tesseract")
    
    def preprocess_image(self, image_data):
        """
        Preprocess image for better OCR accuracy
        """
        try:
            # Convert base64 to image if needed
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding for better text contrast
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Resize image for better OCR (if too small)
            height, width = cleaned.shape
            if height < 100 or width < 100:
                scale_factor = max(200 / height, 200 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Image preprocessing failed: {str(e)}")
            return None
    
    def extract_patient_number(self, image_data):
        """
        Extract patient number from image
        Looks for 10-digit number on 3rd line
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            if processed_image is None:
                return {
                    'success': False,
                    'error': 'Image preprocessing failed'
                }
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)
            
            # Also try with different PSM modes for better results
            text_psm7 = pytesseract.image_to_string(processed_image, config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789')
            text_psm8 = pytesseract.image_to_string(processed_image, config='--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789')
            
            # Combine all OCR results
            all_text = f"{text}\n{text_psm7}\n{text_psm8}"
            
            print(f"🔍 OCR Raw Text: {repr(all_text)}")
            
            # Extract patient numbers (10 digits)
            patient_numbers = self.find_patient_numbers(all_text)
            
            if patient_numbers:
                # Return the first valid patient number found
                return {
                    'success': True,
                    'patient_number': patient_numbers[0],
                    'all_found': patient_numbers,
                    'raw_text': all_text.strip()
                }
            else:
                return {
                    'success': False,
                    'error': 'Geen 10-cijferig patiëntennummer gevonden',
                    'raw_text': all_text.strip(),
                    'suggestion': 'Zorg dat het patiëntennummer duidelijk zichtbaar is in de foto'
                }
                
        except Exception as e:
            print(f"❌ OCR extraction failed: {str(e)}")
            return {
                'success': False,
                'error': f'OCR fout: {str(e)}'
            }
    
    def find_patient_numbers(self, text):
        """
        Find 10-digit patient numbers in text
        """
        # Pattern for exactly 10 digits
        pattern = r'\b\d{10}\b'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
        # Remove duplicates while preserving order
        unique_numbers = []
        for match in matches:
            if match not in unique_numbers:
                unique_numbers.append(match)
        
        return unique_numbers
    
    def extract_from_lines(self, text, target_line=3):
        """
        Extract patient number specifically from the 3rd line
        """
        lines = text.strip().split('\n')
        
        if len(lines) >= target_line:
            third_line = lines[target_line - 1]  # 0-indexed
            patient_numbers = self.find_patient_numbers(third_line)
            if patient_numbers:
                return patient_numbers[0]
        
        # Fallback: search all lines
        return self.find_patient_numbers(text)
    
    def validate_patient_number(self, number):
        """
        Validate patient number format
        """
        if not number:
            return False
        
        # Must be exactly 10 digits
        if not re.match(r'^\d{10}$', str(number)):
            return False
        
        # Additional validation rules can be added here
        # e.g., check digit validation, hospital-specific formats
        
        return True

# Test the OCR service
if __name__ == "__main__":
    ocr = PatientNumberOCR()
    
    # Test with sample text
    test_text = """
    Patient Name: John Doe
    Date of Birth: 01/01/1980
    Patient ID: 1234567890
    Department: Cardiology
    """
    
    numbers = ocr.find_patient_numbers(test_text)
    print(f"Found patient numbers: {numbers}")
    
    # Test validation
    print(f"Valid: {ocr.validate_patient_number('1234567890')}")
    print(f"Invalid: {ocr.validate_patient_number('123456789')}")  # 9 digits
    print(f"Invalid: {ocr.validate_patient_number('12345678901')}")  # 11 digits

