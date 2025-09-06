"""
OCR Service for Patient Number Extraction
Hybrid approach: Local Tesseract + Cloud OpenAI Vision API
Ensures functionality both locally and in cloud deployment
"""

import re
import io
import base64
import os
import requests
from typing import Dict, Any, Optional

# Optional imports for local OCR functionality
try:
    from PIL import Image
    import cv2
    import numpy as np
    import pytesseract
    LOCAL_OCR_AVAILABLE = True
    print("🔍 Local OCR dependencies loaded successfully")
except ImportError as e:
    LOCAL_OCR_AVAILABLE = False
    print(f"⚠️ Local OCR dependencies not available: {e}")

# Check for OpenAI API key for cloud OCR
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CLOUD_OCR_AVAILABLE = bool(OPENAI_API_KEY)

if CLOUD_OCR_AVAILABLE:
    print("🌐 Cloud OCR (OpenAI Vision) available")
else:
    print("⚠️ Cloud OCR not available - no OpenAI API key")

class PatientNumberOCR:
    """
    Hybrid OCR service for extracting patient numbers from photos
    - Local: Uses Tesseract OCR for privacy (no external API calls)
    - Cloud: Uses OpenAI Vision API when Tesseract not available
    - Mobile-friendly: Works on iPhone Safari and other mobile browsers
    """
    
    def __init__(self):
        self.local_available = LOCAL_OCR_AVAILABLE
        self.cloud_available = CLOUD_OCR_AVAILABLE
        self.available = self.local_available or self.cloud_available
        
        if self.local_available:
            # Configure Tesseract for better number recognition
            self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            print("🔍 Local Patient Number OCR initialized with Tesseract")
        
        if self.cloud_available:
            print("🌐 Cloud Patient Number OCR initialized with OpenAI Vision")
        
        if not self.available:
            print("⚠️ No OCR methods available")
            print("💡 Install Tesseract locally OR set OPENAI_API_KEY for cloud OCR")
    
    def is_available(self):
        """Check if any OCR functionality is available"""
        return self.available
    
    def preprocess_image_local(self, image_data):
        """
        Preprocess image for better local OCR accuracy
        """
        if not self.local_available:
            return None
            
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
            print(f"❌ Local image preprocessing failed: {str(e)}")
            return None
    
    def extract_with_openai_vision(self, image_data) -> Dict[str, Any]:
        """
        Extract patient number using OpenAI Vision API
        Perfect for cloud deployment and mobile usage
        """
        if not self.cloud_available:
            return {
                'success': False,
                'error': 'OpenAI Vision API niet beschikbaar - geen API key'
            }
        
        try:
            # Handle both bytes and string input
            if isinstance(image_data, bytes):
                # Convert bytes to base64 string
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
            elif isinstance(image_data, str):
                # Ensure string is in proper data URL format
                if not image_data.startswith('data:image'):
                    image_data_url = f"data:image/jpeg;base64,{image_data}"
                else:
                    image_data_url = image_data
            else:
                return {
                    'success': False,
                    'error': f'Ongeldig image data type: {type(image_data)}'
                }
            
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Look at this medical software interface screenshot and find the patient ID number.

TASK: Extract the patient identification number

WHAT TO LOOK FOR:
- A patient ID number that is typically 9-10 digits long
- Can start with ANY digit (0-9), including numbers starting with 5, 0, 1, etc.
- Examples: 5508201059, 0033001339, 1234567890, 9876543210
- Usually displayed prominently near the patient name and basic info
- This is the main patient identifier in the medical system
- May be labeled as "Patient ID", "ID", or just displayed as a number

TYPICAL LOCATIONS IN MEDICAL SOFTWARE:
- In the patient header section (top area)
- Near patient name and demographic information (name, birth date)
- Could be in a yellow/highlighted field or box
- May be next to a patient photo or photo placeholder
- Often displayed in a larger or bold font
- Usually the longest number sequence visible

WHAT TO IGNORE:
- Phone numbers (usually have spaces, dashes, or different formatting like 0487333549)
- Dates (like 28-08-1955, different format with slashes/dashes)
- Short reference numbers (less than 8 digits)
- Very long numbers (more than 11 digits)
- Numbers that are clearly timestamps or system IDs

RESPONSE FORMAT:
- If you find a 9-10 digit patient ID: return just the number (e.g., "5508201059")
- If not found: respond "NO_PATIENT_ID - I see these numbers: [list all numbers you find]"

Look carefully at the entire interface, especially areas with patient information. The patient ID is usually the most prominent number after the patient's name."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 200,
                "temperature": 0
            }
            
            # Make the API request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result['choices'][0]['message']['content'].strip()
                
                print(f"🌐 OpenAI Vision extracted: {extracted_text}")
                
                # Parse the response
                if "NO_PATIENT_ID" in extracted_text:
                    return {
                        'success': False,
                        'error': 'Geen patiëntennummer gevonden',
                        'raw_text': extracted_text,
                        'suggestion': 'Zorg dat het patiëntennummer (geel vak) duidelijk zichtbaar is',
                        'debug_info': f"AI response: {extracted_text}"
                    }
                
                # Look for patient numbers (9-10 digits) in the response
                patient_numbers = self.find_patient_numbers_flexible(extracted_text)
                if patient_numbers:
                    return {
                        'success': True,
                        'patient_number': patient_numbers[0],
                        'all_found': patient_numbers,
                        'raw_text': extracted_text,
                        'method': 'openai_vision',
                        'debug_info': f"AI found: {extracted_text}"
                    }
                else:
                    # Try to extract any numbers from the response
                    all_numbers = re.findall(r'\d+', extracted_text)
                    valid_numbers = [num for num in all_numbers if len(num) >= 8 and len(num) <= 10]
                    
                    if valid_numbers:
                        # Pad with zeros if needed (for 9-digit numbers)
                        padded_number = valid_numbers[0].zfill(10)
                        return {
                            'success': True,
                            'patient_number': padded_number,
                            'all_found': valid_numbers,
                            'raw_text': extracted_text,
                            'method': 'openai_vision_fallback',
                            'debug_info': f"Fallback extraction: {valid_numbers} -> padded to {padded_number}"
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'AI zag wel tekst maar geen geldig patiëntennummer (8-10 cijfers)',
                            'raw_text': extracted_text,
                            'suggestion': 'Probeer een foto waar het gele vak met patiëntennummer duidelijker zichtbaar is',
                            'debug_info': f"AI response: {extracted_text}, Found numbers: {all_numbers}"
                        }
            else:
                error_text = response.text if response.text else f"HTTP {response.status_code}"
                return {
                    'success': False,
                    'error': f'OpenAI Vision API fout: {response.status_code}',
                    'suggestion': 'Probeer opnieuw of gebruik handmatige invoer',
                    'debug_info': f"API Error: {error_text}"
                }
                
        except Exception as e:
            print(f"❌ OpenAI Vision extraction failed: {str(e)}")
            return {
                'success': False,
                'error': f'Cloud OCR fout: {str(e)}',
                'debug_info': f"Exception: {str(e)}"
            }
    
    def extract_with_local_ocr(self, image_data) -> Dict[str, Any]:
        """
        Extract patient number using local Tesseract OCR
        """
        if not self.local_available:
            return {
                'success': False,
                'error': 'Lokale OCR niet beschikbaar'
            }
            
        try:
            # Preprocess image
            processed_image = self.preprocess_image_local(image_data)
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
            
            print(f"🔍 Local OCR Raw Text: {repr(all_text)}")
            
            # Extract patient numbers (10 digits)
            patient_numbers = self.find_patient_numbers(all_text)
            
            if patient_numbers:
                # Return the first valid patient number found
                return {
                    'success': True,
                    'patient_number': patient_numbers[0],
                    'all_found': patient_numbers,
                    'raw_text': all_text.strip(),
                    'method': 'local_tesseract'
                }
            else:
                return {
                    'success': False,
                    'error': 'Geen 10-cijferig patiëntennummer gevonden',
                    'raw_text': all_text.strip(),
                    'suggestion': 'Zorg dat het patiëntennummer duidelijk zichtbaar is in de foto'
                }
                
        except Exception as e:
            print(f"❌ Local OCR extraction failed: {str(e)}")
            return {
                'success': False,
                'error': f'Lokale OCR fout: {str(e)}'
            }
    
    def extract_patient_number(self, image_data):
        """
        Extract patient number from image using best available method
        Priority: Local OCR (privacy) -> Cloud OCR (compatibility)
        """
        if not self.available:
            return {
                'success': False,
                'error': 'Geen OCR service beschikbaar',
                'suggestion': 'Installeer Tesseract lokaal of configureer OpenAI API key'
            }
        
        # Try local OCR first (better privacy)
        if self.local_available:
            print("🔍 Trying local OCR first...")
            result = self.extract_with_local_ocr(image_data)
            if result['success']:
                return result
            else:
                print(f"⚠️ Local OCR failed: {result['error']}")
        
        # Fallback to cloud OCR
        if self.cloud_available:
            print("🌐 Falling back to cloud OCR...")
            return self.extract_with_openai_vision(image_data)
        
        # No methods available
        return {
            'success': False,
            'error': 'Alle OCR methoden gefaald',
            'suggestion': 'Voer patiëntennummer handmatig in'
        }
    
    def find_patient_numbers_flexible(self, text):
        """
        Find patient numbers (9-10 digits) in text with flexible matching
        Handles numbers with or without leading zeros
        """
        import re
        
        # Clean the text first
        text = text.strip()
        print(f"🔍 Searching for patient numbers in: {repr(text)}")
        
        # Multiple patterns to catch different formats
        patterns = [
            r'\b(\d{10})\b',          # Exactly 10 digits with word boundaries
            r'\b(\d{9})\b',           # Exactly 9 digits with word boundaries  
            r'(\d{10})',              # 10 digits without strict boundaries
            r'(\d{9})',               # 9 digits without strict boundaries
            r'(\d{4}\s*\d{3}\s*\d{3})', # Spaced format: 5508 201 059
            r'(\d{4}\s*\d{6})',       # Spaced format: 5508 201059
        ]
        
        unique_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Remove spaces and validate
                clean_number = re.sub(r'\s+', '', match)
                if clean_number not in unique_numbers and self._is_valid_patient_number(clean_number):
                    unique_numbers.append(clean_number)
                    print(f"✅ Found valid patient number: {clean_number}")
        
        # If no exact matches, try to find any sequence of 9-10 digits
        if not unique_numbers:
            all_digits = re.findall(r'\d+', text)
            print(f"🔍 All digit sequences found: {all_digits}")
            for digits in all_digits:
                if len(digits) >= 9 and len(digits) <= 10 and self._is_valid_patient_number(digits):
                    unique_numbers.append(digits)
                    print(f"✅ Found fallback patient number: {digits}")
        
        return unique_numbers
    
    def _is_valid_patient_number(self, number_str):
        """
        Validate if a number string could be a patient number
        """
        if not number_str or not number_str.isdigit():
            return False
        
        # Must be 9 or 10 digits
        if len(number_str) < 9 or len(number_str) > 10:
            return False
        
        # Reject obvious non-patient numbers
        invalid_patterns = [
            r'^0{5,}',          # Too many leading zeros (00000...)
            r'^1{5,}',          # Too many ones (11111...)
            r'^9{5,}',          # Too many nines (99999...)
            r'^(19|20)\d{8}$',  # Looks like a timestamp (19xxxxxxxx, 20xxxxxxxx)
            r'^123456789[0-9]?$',  # Sequential test numbers
            r'^987654321[0-9]?$',  # Sequential test numbers
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, number_str):
                return False
        
        return True
    
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
    
    print(f"Local OCR available: {ocr.local_available}")
    print(f"Cloud OCR available: {ocr.cloud_available}")
    print(f"Any OCR available: {ocr.is_available()}")
    
    if ocr.local_available:
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
    else:
        print("Local OCR functionality not available - install dependencies for testing")

