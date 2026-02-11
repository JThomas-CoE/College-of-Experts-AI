"""
Template Validator - V12 Quality Gate

Implements the 7-tier validation system for task frameworks:

ERROR-LEVEL (fails adaptation):
  E1: Missing required fields (id, title, description, slots)
  E2: Invalid slot ID format (must be snake_case identifier)
  E3: Duplicate slot IDs
  E4: Referenced persona not in registry
  E5: Cyclic dependencies (fails DAG check)

WARNING-LEVEL (logs, but passes):
  W1: Empty dependencies list (isolated slot)
  W2: Slot with >3 dependencies (potential bottleneck)

Uses TaskFramework.validate() for cycle detection.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import TaskFramework for structural validation
from src.framework_scheduler import TaskFramework, FrameworkSlot

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    ERROR = "error"      # Fails validation
    WARNING = "warning"  # Logged but passes


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    code: str                      # E1, E2, W1, etc.
    severity: ValidationSeverity
    message: str
    slot_id: Optional[str] = None  # Which slot caused the issue, if applicable
    
    def __str__(self):
        loc = f" (slot: {self.slot_id})" if self.slot_id else ""
        return f"[{self.code}] {self.message}{loc}"


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def all_issues(self) -> List[ValidationIssue]:
        """All issues sorted by severity."""
        return self.errors + self.warnings
    
    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_valid:
            status = "✓ VALID"
        else:
            status = "✗ INVALID"
        
        parts = [f"{status} | {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        
        for issue in self.errors:
            parts.append(f"  ERROR {issue}")
        
        for issue in self.warnings:
            parts.append(f"  WARN  {issue}")
        
        return "\n".join(parts)


class TemplateValidator:
    """
    Validates task framework templates against the V12 quality gate.
    
    Supports validation of:
    - Raw dictionaries (from JSON or adapter output)
    - TaskFramework objects
    """
    
    # Valid slot ID pattern: snake_case identifier
    SLOT_ID_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
    
    # Required top-level fields
    REQUIRED_FIELDS = {'id', 'title', 'description', 'slots'}
    
    # Required slot fields
    REQUIRED_SLOT_FIELDS = {'id', 'title'}
    
    # Optional slot fields that should be validated
    OPTIONAL_SLOT_FIELDS = {'description', 'persona', 'assigned_expert', 'expected_outputs', 'dependencies', 'can_reference'}
    
    def __init__(self, persona_registry_path: Optional[Path] = None):
        """
        Initialize validator.
        
        Args:
            persona_registry_path: Path to index.json for persona validation.
                                   If None, E4 checks are skipped.
        """
        self.persona_ids: Set[str] = set()
        
        if persona_registry_path and persona_registry_path.exists():
            self._load_persona_registry(persona_registry_path)
    
    def _load_persona_registry(self, path: Path):
        """Load valid persona IDs from registry."""
        try:
            with open(path) as f:
                registry = json.load(f)
            
            if isinstance(registry, dict) and "personas" in registry:
                self.persona_ids = set(registry["personas"].keys())
            else:
                # Assume it's a flat dict of persona_id -> definition
                self.persona_ids = set(registry.keys())
            
            logger.info(f"Loaded {len(self.persona_ids)} persona IDs from registry")
        except Exception as e:
            logger.warning(f"Failed to load persona registry: {e}")
    
    def validate(self, template: Dict[str, Any] | TaskFramework) -> ValidationResult:
        """
        Validate a template.
        
        Args:
            template: Dictionary or TaskFramework to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []
        
        # Convert to dict if needed
        if isinstance(template, TaskFramework):
            template_dict = template.to_dict()
        else:
            template_dict = template
        
        # E1: Check required fields
        missing_fields = self._check_required_fields(template_dict)
        for field in missing_fields:
            errors.append(ValidationIssue(
                code="E1",
                severity=ValidationSeverity.ERROR,
                message=f"Missing required field: {field}"
            ))
        
        # Can't continue if basic structure is missing
        if missing_fields:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        slots = template_dict.get("slots", [])
        
        # Check slot-level issues
        slot_ids: Set[str] = set()
        
        for slot in slots:
            slot_id = slot.get("id", "")
            
            # E2: Invalid slot ID format
            if not self.SLOT_ID_PATTERN.match(slot_id):
                errors.append(ValidationIssue(
                    code="E2",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid slot ID format '{slot_id}' (must be snake_case)",
                    slot_id=slot_id
                ))
            
            # E3: Duplicate slot ID
            if slot_id in slot_ids:
                errors.append(ValidationIssue(
                    code="E3",
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate slot ID: {slot_id}",
                    slot_id=slot_id
                ))
            slot_ids.add(slot_id)
            
            # E4: Invalid persona reference (if persona registry loaded)
            if self.persona_ids:
                assigned_persona = slot.get("assigned_expert") or slot.get("persona")
                if assigned_persona:
                    if assigned_persona not in self.persona_ids:
                        errors.append(ValidationIssue(
                            code="E4",
                            severity=ValidationSeverity.ERROR,
                            message=f"Unknown persona reference: {assigned_persona}",
                            slot_id=slot_id
                        ))
                else:
                    # W3: No persona assigned when registry is loaded
                    warnings.append(ValidationIssue(
                        code="W3",
                        severity=ValidationSeverity.WARNING,
                        message="Slot has no persona assignment (recommended for persona-aware routing)",
                        slot_id=slot_id
                    ))
            
            # W1: Empty dependencies
            deps = slot.get("dependencies", [])
            if not deps and len(slots) > 1:
                # Only warn if there are multiple slots (single-slot frameworks are fine)
                warnings.append(ValidationIssue(
                    code="W1",
                    severity=ValidationSeverity.WARNING,
                    message="Slot has no dependencies (isolated)",
                    slot_id=slot_id
                ))
            
            # W2: Too many dependencies
            if len(deps) > 3:
                warnings.append(ValidationIssue(
                    code="W2",
                    severity=ValidationSeverity.WARNING,
                    message=f"Slot has {len(deps)} dependencies (potential bottleneck)",
                    slot_id=slot_id
                ))
        
        # E5: Cyclic dependencies (use TaskFramework.validate())
        cycle_errors = self._check_cycles(template_dict)
        errors.extend(cycle_errors)
        
        # Determine overall validity
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
        # Log result
        if is_valid:
            logger.debug(f"Template '{template_dict.get('id', 'unknown')}' passed validation")
        else:
            logger.warning(f"Template '{template_dict.get('id', 'unknown')}' failed validation:\n{result.summary()}")
        
        return result
    
    def _check_required_fields(self, template: Dict[str, Any]) -> List[str]:
        """Check for missing required fields."""
        missing = []
        
        for field in self.REQUIRED_FIELDS:
            if field not in template:
                missing.append(field)
            elif field == "slots":
                # Slots must be a non-empty list
                slots = template.get("slots")
                if not isinstance(slots, list):
                    missing.append("slots (must be a list)")
                elif len(slots) == 0:
                    missing.append("slots (must be non-empty)")
        
        return missing
    
    def _check_cycles(self, template: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for cyclic dependencies using TaskFramework."""
        errors = []
        
        try:
            # Build TaskFramework for validation
            framework = TaskFramework(
                id=template.get("id", "temp"),
                title=template.get("title", ""),
                description=template.get("description", "")
            )
            
            for slot_data in template.get("slots", []):
                framework.add_slot(FrameworkSlot(
                    id=slot_data.get("id", ""),
                    title=slot_data.get("title", ""),
                    description=slot_data.get("description", ""),
                    dependencies=slot_data.get("dependencies", []),
                    can_reference=slot_data.get("can_reference", [])
                ))
            
            # Use built-in validation
            validation_errors = framework.validate()
            
            for error in validation_errors:
                if "circular" in error.lower() or "cycle" in error.lower():
                    errors.append(ValidationIssue(
                        code="E5",
                        severity=ValidationSeverity.ERROR,
                        message=error
                    ))
                elif "unknown slot" in error.lower():
                    # This catches invalid dependency references
                    errors.append(ValidationIssue(
                        code="E5",
                        severity=ValidationSeverity.ERROR,
                        message=error
                    ))
        except Exception as e:
            logger.error(f"Error during cycle check: {e}")
            errors.append(ValidationIssue(
                code="E5",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate DAG structure: {e}"
            ))
        
        return errors
    
    def validate_batch(self, templates: List[Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """
        Validate multiple templates.
        
        Args:
            templates: List of template dictionaries
            
        Returns:
            Dict mapping template ID to ValidationResult
        """
        results = {}
        
        for template in templates:
            template_id = template.get("id", f"unnamed_{len(results)}")
            results[template_id] = self.validate(template)
        
        return results


def validate_template_file(file_path: Path, persona_registry_path: Optional[Path] = None) -> ValidationResult:
    """
    Convenience function to validate a template file.
    
    Args:
        file_path: Path to template JSON file
        persona_registry_path: Path to persona registry
        
    Returns:
        ValidationResult
    """
    with open(file_path) as f:
        template = json.load(f)
    
    validator = TemplateValidator(persona_registry_path)
    return validator.validate(template)


if __name__ == "__main__":
    # Test the validator
    logging.basicConfig(level=logging.DEBUG)
    
    # Valid template
    valid_template = {
        "id": "test_template_001",
        "title": "Test Template",
        "description": "A test template for validation",
        "slots": [
            {"id": "slot_a", "title": "Slot A", "dependencies": []},
            {"id": "slot_b", "title": "Slot B", "dependencies": ["slot_a"]},
            {"id": "slot_c", "title": "Slot C", "dependencies": ["slot_a", "slot_b"]}
        ]
    }
    
    # Invalid template (cycle)
    cyclic_template = {
        "id": "cyclic_template",
        "title": "Cyclic Template",
        "description": "Has a cycle",
        "slots": [
            {"id": "slot_x", "title": "Slot X", "dependencies": ["slot_y"]},
            {"id": "slot_y", "title": "Slot Y", "dependencies": ["slot_x"]}
        ]
    }
    
    # Invalid template (bad ID format)
    bad_id_template = {
        "id": "bad_id_template",
        "title": "Bad ID Template",
        "description": "Has invalid slot IDs",
        "slots": [
            {"id": "SlotA", "title": "Slot A", "dependencies": []},  # CamelCase
            {"id": "slot-b", "title": "Slot B", "dependencies": []}  # Contains dash
        ]
    }
    
    validator = TemplateValidator()
    
    print("=" * 60)
    print("Template Validator Test")
    print("=" * 60)
    
    print("\n1. Valid Template:")
    result = validator.validate(valid_template)
    print(result.summary())
    
    print("\n2. Cyclic Template:")
    result = validator.validate(cyclic_template)
    print(result.summary())
    
    print("\n3. Bad ID Template:")
    result = validator.validate(bad_id_template)
    print(result.summary())
