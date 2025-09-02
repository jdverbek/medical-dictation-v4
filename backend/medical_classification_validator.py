"""
Medical Classification Validator
Prevents critical medical classification errors like LVEF misclassification
"""

import re
from typing import Dict, List, Tuple, Optional

class MedicalClassificationValidator:
    """
    Validates medical classifications to prevent dangerous errors
    """
    
    def __init__(self):
        # ESC 2023 Heart Failure Guidelines - CRITICAL CLASSIFICATIONS
        self.hf_classifications = {
            'HFrEF': {'min_lvef': 0, 'max_lvef': 40, 'description': 'Heart Failure with reduced Ejection Fraction'},
            'HFmrEF': {'min_lvef': 41, 'max_lvef': 49, 'description': 'Heart Failure with mildly reduced Ejection Fraction'},
            'HFpEF': {'min_lvef': 50, 'max_lvef': 100, 'description': 'Heart Failure with preserved Ejection Fraction'}
        }
        
        # Aortic stenosis classifications (ESC 2021)
        self.as_classifications = {
            'mild': {'max_gradient': 19, 'max_velocity': 2.9},
            'moderate': {'min_gradient': 20, 'max_gradient': 39, 'min_velocity': 3.0, 'max_velocity': 3.9},
            'severe': {'min_gradient': 40, 'min_velocity': 4.0}
        }
        
        # Mitral regurgitation classifications
        self.mr_classifications = {
            'mild': {'max_vc': 2.9, 'max_ero': 19, 'max_rvol': 29},
            'moderate': {'min_vc': 3.0, 'max_vc': 6.9, 'min_ero': 20, 'max_ero': 39, 'min_rvol': 30, 'max_rvol': 59},
            'severe': {'min_vc': 7.0, 'min_ero': 40, 'min_rvol': 60}
        }
        
        # LA dimension classifications
        self.la_classifications = {
            'normal': {'max_diameter': 40},
            'mildly_dilated': {'min_diameter': 41, 'max_diameter': 46},
            'moderately_dilated': {'min_diameter': 47, 'max_diameter': 51},
            'severely_dilated': {'min_diameter': 52}
        }
        
        # CVD correlations
        self.cvd_correlations = {
            'collapsed': {'cvd_range': (0, 5), 'ivc_size': '<17mm', 'variability': '>50%'},
            'normal': {'cvd_range': (5, 10), 'ivc_size': '17-21mm', 'variability': '50%'},
            'elevated': {'cvd_range': (10, 15), 'ivc_size': '>17mm', 'variability': '<50%'},
            'high': {'cvd_range': (15, 20), 'ivc_size': '>21mm', 'variability': '<50%'}
        }
    
    def validate_lvef_classification(self, text: str) -> List[Dict[str, str]]:
        """
        Validate LVEF classifications in medical text
        Returns list of validation errors
        """
        errors = []
        
        # Pattern to find LVEF values and associated classifications
        lvef_patterns = [
            r'LVEF\s*(?:van\s*)?(\d+)(?:-(\d+))?%?\s*.*?(HFrEF|HFmrEF|HFpEF|verminderde?\s*functie|gereduceerde?\s*functie|bewaarde?\s*functie)',
            r'(HFrEF|HFmrEF|HFpEF|verminderde?\s*functie|gereduceerde?\s*functie|bewaarde?\s*functie).*?LVEF\s*(?:van\s*)?(\d+)(?:-(\d+))?%?',
            r'LVEF\s*(?:van\s*)?(\d+)(?:-(\d+))?%?\s*.*?(verminderde?\s*linker\s*ventrikelfunctie|gereduceerde?\s*linker\s*ventrikelfunctie)',
            r'(\d+)(?:-(\d+))?%?\s*.*?(verminderde?\s*functie|gereduceerde?\s*functie).*?LVEF',
            r'linker\s*ventrikelfunctie.*?LVEF\s*(?:van\s*)?(\d+)(?:-(\d+))?%?\s*.*?(verminderde?|gereduceerde?|bewaarde?)',
            r'verminderde?\s*linker\s*ventrikelfunctie.*?LVEF\s*(?:van\s*)?(\d+)(?:-(\d+))?%?\s*.*?(HFrEF|HFmrEF|HFpEF)',
            r'(verminderde?\s*linker\s*ventrikelfunctie).*?LVEF\s*(?:van\s*)?(\d+)(?:-(\d+))?%?'
        ]
        
        for pattern in lvef_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                # Extract LVEF value(s) and classification
                if groups[0] and groups[0].isdigit():
                    # Pattern: LVEF first, then classification
                    lvef_min = int(groups[0])
                    lvef_max = int(groups[1]) if groups[1] and groups[1].isdigit() else lvef_min
                    classification = groups[2] if len(groups) > 2 else ""
                elif groups[1] and groups[1].isdigit():
                    # Pattern: classification first, then LVEF
                    classification = groups[0]
                    lvef_min = int(groups[1])
                    lvef_max = int(groups[2]) if len(groups) > 2 and groups[2] and groups[2].isdigit() else lvef_min
                else:
                    continue
                
                # Use average for range
                lvef_value = (lvef_min + lvef_max) / 2
                
                # Validate classification
                error = self._validate_single_lvef_classification(lvef_value, classification, match.group())
                if error:
                    errors.append(error)
        
        return errors
    
    def _validate_single_lvef_classification(self, lvef_value: float, classification: str, original_text: str) -> Optional[Dict[str, str]]:
        """
        Validate a single LVEF classification
        """
        classification_lower = classification.lower()
        
        # Determine correct classification based on LVEF value
        correct_classification = None
        for hf_type, ranges in self.hf_classifications.items():
            if ranges['min_lvef'] <= lvef_value <= ranges['max_lvef']:
                correct_classification = hf_type
                break
        
        # Check for specific errors
        if 'hfref' in classification_lower and lvef_value > 40:
            return {
                'type': 'CRITICAL_LVEF_ERROR',
                'issue': f'LVEF {lvef_value}% geclassificeerd als HFrEF',
                'correction': f'LVEF {lvef_value}% is HFmrEF (41-49%) of HFpEF (≥50%), NIET HFrEF (≤40%)',
                'severity': 'LEVENSGEVAARLIJK',
                'original_text': original_text,
                'esc_guideline': 'ESC 2023 Heart Failure Guidelines'
            }
        
        if 'hfmref' in classification_lower and (lvef_value <= 40 or lvef_value >= 50):
            return {
                'type': 'CRITICAL_LVEF_ERROR',
                'issue': f'LVEF {lvef_value}% geclassificeerd als HFmrEF',
                'correction': f'LVEF {lvef_value}% is {"HFrEF (≤40%)" if lvef_value <= 40 else "HFpEF (≥50%)"}, NIET HFmrEF (41-49%)',
                'severity': 'ERNSTIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2023 Heart Failure Guidelines'
            }
        
        if 'hfpef' in classification_lower and lvef_value < 50:
            return {
                'type': 'CRITICAL_LVEF_ERROR',
                'issue': f'LVEF {lvef_value}% geclassificeerd als HFpEF',
                'correction': f'LVEF {lvef_value}% is {"HFrEF (≤40%)" if lvef_value <= 40 else "HFmrEF (41-49%)"}, NIET HFpEF (≥50%)',
                'severity': 'ERNSTIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2023 Heart Failure Guidelines'
            }
        
        # Check for descriptive terms
        if ('verminderde' in classification_lower or 'gereduceerde' in classification_lower) and lvef_value >= 50:
            return {
                'type': 'CRITICAL_LVEF_ERROR',
                'issue': f'LVEF {lvef_value}% beschreven als "verminderde functie"',
                'correction': f'LVEF {lvef_value}% is normale/bewaarde functie (≥50%), NIET verminderd',
                'severity': 'ERNSTIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2023 Heart Failure Guidelines'
            }
        
        if 'bewaarde' in classification_lower and lvef_value < 50:
            return {
                'type': 'CRITICAL_LVEF_ERROR',
                'issue': f'LVEF {lvef_value}% beschreven als "bewaarde functie"',
                'correction': f'LVEF {lvef_value}% is {"ernstig verminderd (≤40%)" if lvef_value <= 40 else "licht verminderd (41-49%)"}, NIET bewaarde functie',
                'severity': 'ERNSTIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2023 Heart Failure Guidelines'
            }
        
        return None
    
    def validate_la_dimensions(self, text: str) -> List[Dict[str, str]]:
        """
        Validate LA dimension classifications
        """
        errors = []
        
        # Pattern to find LA dimensions and descriptions
        la_patterns = [
            r'LA\s*(\d+)\s*mm.*?(normaal|licht\s*gedilateerd|matig\s*gedilateerd|sterk\s*gedilateerd)',
            r'(normaal|licht\s*gedilateerd|matig\s*gedilateerd|sterk\s*gedilateerd).*?LA\s*(\d+)\s*mm',
            r'linker\s*atrium.*?(\d+)\s*mm.*?(normaal|licht\s*gedilateerd|matig\s*gedilateerd|sterk\s*gedilateerd)'
        ]
        
        for pattern in la_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                # Extract LA dimension and description
                if groups[0] and groups[0].isdigit():
                    la_size = int(groups[0])
                    description = groups[1]
                elif groups[1] and groups[1].isdigit():
                    description = groups[0]
                    la_size = int(groups[1])
                else:
                    continue
                
                # Validate classification
                error = self._validate_la_classification(la_size, description, match.group())
                if error:
                    errors.append(error)
        
        return errors
    
    def _validate_la_classification(self, la_size: int, description: str, original_text: str) -> Optional[Dict[str, str]]:
        """
        Validate LA dimension classification
        """
        description_lower = description.lower()
        
        # Determine correct classification
        if la_size <= 40:
            correct = "normaal"
        elif la_size <= 46:
            correct = "licht gedilateerd"
        elif la_size <= 51:
            correct = "matig gedilateerd"
        else:
            correct = "sterk gedilateerd"
        
        # Check for errors
        if 'normaal' in description_lower and la_size > 40:
            return {
                'type': 'LA_DIMENSION_ERROR',
                'issue': f'LA {la_size}mm beschreven als "normaal"',
                'correction': f'LA {la_size}mm is {correct} (normaal ≤40mm)',
                'severity': 'MATIG',
                'original_text': original_text
            }
        
        if 'licht gedilateerd' in description_lower and (la_size <= 40 or la_size > 46):
            return {
                'type': 'LA_DIMENSION_ERROR',
                'issue': f'LA {la_size}mm beschreven als "licht gedilateerd"',
                'correction': f'LA {la_size}mm is {correct} (licht gedilateerd 41-46mm)',
                'severity': 'MATIG',
                'original_text': original_text
            }
        
        return None
    
    def validate_cvd_correlations(self, text: str) -> List[Dict[str, str]]:
        """
        Validate CVD correlations with vena cava descriptions
        """
        errors = []
        
        # Pattern to find vena cava descriptions and CVD values
        cvd_patterns = [
            r'vena\s*cava.*?(plat|collapsed|stuwing|gedilateerd).*?CVD\s*(\d+)(?:-(\d+))?\s*mmHg',
            r'CVD\s*(\d+)(?:-(\d+))?\s*mmHg.*?vena\s*cava.*?(plat|collapsed|stuwing|gedilateerd)',
            r'vena\s*cava.*?(\d+)\s*mm.*?(plat|collapsed|stuwing|gedilateerd).*?CVD\s*(\d+)(?:-(\d+))?\s*mmHg'
        ]
        
        for pattern in cvd_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                # Extract information based on pattern
                if len(groups) >= 3:
                    if groups[0] and groups[0].isdigit():
                        # CVD first pattern
                        cvd_min = int(groups[0])
                        cvd_max = int(groups[1]) if groups[1] and groups[1].isdigit() else cvd_min
                        description = groups[2]
                    elif groups[2] and groups[2].isdigit():
                        # Vena cava first pattern
                        description = groups[0]
                        cvd_min = int(groups[2])
                        cvd_max = int(groups[3]) if len(groups) > 3 and groups[3] and groups[3].isdigit() else cvd_min
                    else:
                        continue
                    
                    cvd_value = (cvd_min + cvd_max) / 2
                    
                    # Validate correlation
                    error = self._validate_cvd_correlation(cvd_value, description, match.group())
                    if error:
                        errors.append(error)
        
        return errors
    
    def _validate_cvd_correlation(self, cvd_value: float, description: str, original_text: str) -> Optional[Dict[str, str]]:
        """
        Validate CVD correlation with vena cava description
        """
        description_lower = description.lower()
        
        # Check for contradictions
        if ('plat' in description_lower or 'collapsed' in description_lower) and cvd_value > 5:
            return {
                'type': 'CVD_CORRELATION_ERROR',
                'issue': f'Vena cava "plat" met CVD {cvd_value}mmHg',
                'correction': f'Vena cava plat impliceert CVD 0-5mmHg, niet {cvd_value}mmHg',
                'severity': 'MATIG',
                'original_text': original_text,
                'explanation': 'Platte vena cava: <17mm diameter, >50% ademvariatie, CVD 0-5mmHg'
            }
        
        if 'stuwing' in description_lower and cvd_value < 10:
            return {
                'type': 'CVD_CORRELATION_ERROR',
                'issue': f'Vena cava "stuwing" met CVD {cvd_value}mmHg',
                'correction': f'Vena cava stuwing impliceert CVD 10-15mmHg, niet {cvd_value}mmHg',
                'severity': 'MATIG',
                'original_text': original_text,
                'explanation': 'Vena cava stuwing: >17mm diameter, <50% ademvariatie, CVD 10-15mmHg'
            }
        
        return None
    
    def validate_aortic_stenosis(self, text: str) -> List[Dict[str, str]]:
        """
        Validate aortic stenosis classifications
        """
        errors = []
        
        # Pattern to find AS gradients and classifications
        as_patterns = [
            r'aorta.*?stenose.*?(mild|matig|ernstig).*?gradient\s*(\d+)\s*mmHg',
            r'gradient\s*(\d+)\s*mmHg.*?aorta.*?stenose.*?(mild|matig|ernstig)',
            r'aorta.*?(mild|matig|ernstig).*?stenose.*?gradient\s*(\d+)\s*mmHg'
        ]
        
        for pattern in as_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 2:
                    if groups[0] and groups[0].isdigit():
                        gradient = int(groups[0])
                        severity = groups[1]
                    elif groups[1] and groups[1].isdigit():
                        severity = groups[0]
                        gradient = int(groups[1])
                    else:
                        continue
                    
                    # Validate classification
                    error = self._validate_as_classification(gradient, severity, match.group())
                    if error:
                        errors.append(error)
        
        return errors
    
    def _validate_as_classification(self, gradient: int, severity: str, original_text: str) -> Optional[Dict[str, str]]:
        """
        Validate aortic stenosis classification
        """
        severity_lower = severity.lower()
        
        # Determine correct classification
        if gradient < 20:
            correct = "mild"
        elif gradient < 40:
            correct = "matig"
        else:
            correct = "ernstig"
        
        # Check for errors
        if 'mild' in severity_lower and gradient >= 20:
            return {
                'type': 'AS_CLASSIFICATION_ERROR',
                'issue': f'Aortastenose gradient {gradient}mmHg geclassificeerd als "mild"',
                'correction': f'Gradient {gradient}mmHg is {correct} stenose (mild <20mmHg)',
                'severity': 'MATIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2021 Valvular Heart Disease Guidelines'
            }
        
        if 'matig' in severity_lower and (gradient < 20 or gradient >= 40):
            return {
                'type': 'AS_CLASSIFICATION_ERROR',
                'issue': f'Aortastenose gradient {gradient}mmHg geclassificeerd als "matig"',
                'correction': f'Gradient {gradient}mmHg is {correct} stenose (matig 20-39mmHg)',
                'severity': 'MATIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2021 Valvular Heart Disease Guidelines'
            }
        
        if 'ernstig' in severity_lower and gradient < 40:
            return {
                'type': 'AS_CLASSIFICATION_ERROR',
                'issue': f'Aortastenose gradient {gradient}mmHg geclassificeerd als "ernstig"',
                'correction': f'Gradient {gradient}mmHg is {correct} stenose (ernstig ≥40mmHg)',
                'severity': 'MATIG',
                'original_text': original_text,
                'esc_guideline': 'ESC 2021 Valvular Heart Disease Guidelines'
            }
        
        return None
    
    def validate_all(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Run all validation checks on medical text
        """
        return {
            'lvef_errors': self.validate_lvef_classification(text),
            'la_errors': self.validate_la_dimensions(text),
            'cvd_errors': self.validate_cvd_correlations(text),
            'as_errors': self.validate_aortic_stenosis(text)
        }
    
    def format_validation_report(self, validation_results: Dict[str, List[Dict[str, str]]]) -> str:
        """
        Format validation results into a readable report
        """
        report_lines = []
        
        # Count total errors
        total_errors = sum(len(errors) for errors in validation_results.values())
        
        if total_errors == 0:
            return "✅ QUALITY CONTROL: Geen medische classificatie fouten gedetecteerd."
        
        report_lines.append(f"🚨 QUALITY CONTROL: {total_errors} medische classificatie fout(en) gedetecteerd:")
        report_lines.append("")
        
        # LVEF errors (most critical)
        if validation_results['lvef_errors']:
            report_lines.append("🔴 KRITIEKE LVEF CLASSIFICATIE FOUTEN:")
            for error in validation_results['lvef_errors']:
                report_lines.append(f"   • {error['issue']}")
                report_lines.append(f"     CORRECTIE: {error['correction']}")
                report_lines.append(f"     ERNST: {error['severity']}")
                if 'esc_guideline' in error:
                    report_lines.append(f"     RICHTLIJN: {error['esc_guideline']}")
                report_lines.append("")
        
        # LA dimension errors
        if validation_results['la_errors']:
            report_lines.append("🟡 LA DIMENSIE FOUTEN:")
            for error in validation_results['la_errors']:
                report_lines.append(f"   • {error['issue']}")
                report_lines.append(f"     CORRECTIE: {error['correction']}")
                report_lines.append("")
        
        # CVD correlation errors
        if validation_results['cvd_errors']:
            report_lines.append("🟡 CVD CORRELATIE FOUTEN:")
            for error in validation_results['cvd_errors']:
                report_lines.append(f"   • {error['issue']}")
                report_lines.append(f"     CORRECTIE: {error['correction']}")
                if 'explanation' in error:
                    report_lines.append(f"     UITLEG: {error['explanation']}")
                report_lines.append("")
        
        # AS classification errors
        if validation_results['as_errors']:
            report_lines.append("🟡 AORTASTENOSE CLASSIFICATIE FOUTEN:")
            for error in validation_results['as_errors']:
                report_lines.append(f"   • {error['issue']}")
                report_lines.append(f"     CORRECTIE: {error['correction']}")
                if 'esc_guideline' in error:
                    report_lines.append(f"     RICHTLIJN: {error['esc_guideline']}")
                report_lines.append("")
        
        return "\n".join(report_lines)

# Test the validator
if __name__ == "__main__":
    validator = MedicalClassificationValidator()
    
    # Test LVEF classification error
    test_text = "De patiënt vertoont een verminderde linker ventrikelfunctie met een LVEF van 40-45%, wat wijst op HFrEF."
    
    results = validator.validate_all(test_text)
    report = validator.format_validation_report(results)
    
    print("TEST VALIDATION REPORT:")
    print(report)

