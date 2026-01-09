"""
Symbolic State Space (Σ) - Unified World Representation

Maintains a typed key-value mapping representing trusted world information during reasoning.
"""

from typing import Dict, Any, Optional, List, Set, Union
from enum import Enum
from dataclasses import dataclass, field
import json


class StateType(Enum):
    """Type enumeration for state fields"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    # Domain-specific types
    CITY = "City"
    MOVIE = "Movie"
    CINEMA = "Cinema"
    SHOWTIME = "Showtime"
    ORDER = "Order"
    TICKET = "Ticket"
    USER_ID = "UserID"
    # Optional type (represented by ?)
    OPTIONAL = "optional"


@dataclass
class StateField:
    """Definition of state field"""
    name: str
    value: Any
    type: StateType
    optional: bool = False
    constraints: List[str] = field(default_factory=list)  # Constraint list


class SymbolicState:
    """
    Symbolic State Space
    
    Maintains a typed key-value mapping supporting:
    - Type checking
    - Constraint validation
    - State update operations
    """
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        Initialize symbolic state
        
        Args:
            initial_state: Initial state dictionary in format {field_name: (value, type_str)}
        """
        self._fields: Dict[str, StateField] = {}
        self._constraints: List[str] = []  # Global constraints
        
        if initial_state:
            for name, value_info in initial_state.items():
                if isinstance(value_info, tuple):
                    value, type_str = value_info
                    state_type = self._parse_type(type_str)
                else:
                    value = value_info
                    state_type = StateType.STRING  # Default type
                
                self.set_field(name, value, state_type)
    
    def _parse_type(self, type_str: str) -> StateType:
        """Parse type string"""
        type_str = type_str.strip()
        if type_str.endswith('?'):
            # Optional type
            base_type = type_str[:-1].strip()
            return StateType.OPTIONAL
        
        type_mapping = {
            'str': StateType.STRING,
            'string': StateType.STRING,
            'int': StateType.INTEGER,
            'integer': StateType.INTEGER,
            'float': StateType.FLOAT,
            'bool': StateType.BOOLEAN,
            'boolean': StateType.BOOLEAN,
            'list': StateType.LIST,
            'dict': StateType.DICT,
        }
        
        # Try direct enum matching
        try:
            return StateType[type_str.upper()]
        except KeyError:
            pass
        
        # Try mapping
        return type_mapping.get(type_str.lower(), StateType.STRING)
    
    def set_field(self, name: str, value: Any, field_type: StateType, 
                  optional: bool = False, constraints: Optional[List[str]] = None):
        """
        Set state field
        
        Args:
            name: Field name
            value: Field value
            field_type: Field type
            optional: Whether optional
            constraints: Constraint list
        """
        self._fields[name] = StateField(
            name=name,
            value=value,
            type=field_type,
            optional=optional,
            constraints=constraints or []
        )
    
    def get_field(self, name: str) -> Optional[Any]:
        """Get field value"""
        if name in self._fields:
            return self._fields[name].value
        return None
    
    def has_field(self, name: str) -> bool:
        """Check if field exists"""
        return name in self._fields
    
    def exists(self, name: str) -> bool:
        """Check if field exists and is non-empty (satisfies precondition check)"""
        if name not in self._fields:
            return False
        field = self._fields[name]
        if field.optional and field.value is None:
            return False
        return field.value is not None
    
    def update(self, updates: Dict[str, Any]) -> 'SymbolicState':
        """
        Update state (returns new state, does not modify original)
        
        Args:
            updates: Update dictionary in format {field_name: value}
        
        Returns:
            New SymbolicState object
        """
        new_state = SymbolicState()
        # Copy existing fields
        for name, field in self._fields.items():
            new_state._fields[name] = StateField(
                name=field.name,
                value=field.value,
                type=field.type,
                optional=field.optional,
                constraints=field.constraints.copy()
            )
        
        # Apply updates
        for name, value in updates.items():
            if name in new_state._fields:
                # Update existing field
                field = new_state._fields[name]
                new_state._fields[name] = StateField(
                    name=field.name,
                    value=value,
                    type=field.type,
                    optional=field.optional,
                    constraints=field.constraints.copy()
                )
            else:
                # Add new field (default type is STRING)
                new_state.set_field(name, value, StateType.STRING)
        
        new_state._constraints = self._constraints.copy()
        return new_state
    
    def merge(self, other: 'SymbolicState') -> 'SymbolicState':
        """Merge two states"""
        new_state = SymbolicState()
        # Copy current state
        for name, field in self._fields.items():
            new_state._fields[name] = StateField(
                name=field.name,
                value=field.value,
                type=field.type,
                optional=field.optional,
                constraints=field.constraints.copy()
            )
        # Merge other state
        for name, field in other._fields.items():
            new_state._fields[name] = StateField(
                name=field.name,
                value=field.value,
                type=field.type,
                optional=field.optional,
                constraints=field.constraints.copy()
            )
        new_state._constraints = self._constraints + other._constraints
        return new_state
    
    def add_constraint(self, constraint: str):
        """Add global constraint"""
        self._constraints.append(constraint)
    
    def get_summary(self) -> str:
        """Get text summary of state (for LLM reasoning)"""
        summary_parts = []
        for name, field in self._fields.items():
            if field.value is not None:
                value_str = str(field.value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                summary_parts.append(f"{name}({field.type.value}): {value_str}")
        
        if summary_parts:
            return "Current State: " + "; ".join(summary_parts)
        return "Current State: Empty"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            name: {
                "value": field.value,
                "type": field.type.value,
                "optional": field.optional,
                "constraints": field.constraints
            }
            for name, field in self._fields.items()
        }
    
    def __repr__(self) -> str:
        return f"SymbolicState(fields={len(self._fields)}, constraints={len(self._constraints)})"
    
    def __str__(self) -> str:
        return self.get_summary()

