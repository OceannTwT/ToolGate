"""
Tool Contract Definition

Each tool t corresponds to a contract Contract(t) = (P_t, Q_t)
- P_t: Precondition, describes the state in which the tool can be called
- Q_t: Postcondition, describes the constraints that must be satisfied after calling
"""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Precondition:
    """
    Precondition P_t
    
    Defines a predicate on the state space Σ that determines whether state S satisfies the calling condition
    """
    # List of constraints in DSL format, e.g., ["exists(city)", "exists(movie)"]
    constraints: List[str] = field(default_factory=list)
    
    # Optional Python function (for complex logic)
    check_function: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def check(self, state: 'SymbolicState') -> bool:
        """
        Check if state satisfies precondition
        
        Args:
            state: Symbolic state object
        
        Returns:
            True if satisfied, False otherwise
        """
        # If custom function exists, use it first
        if self.check_function:
            try:
                return self.check_function(state.to_dict())
            except Exception:
                return False
        
        # Otherwise parse DSL constraints
        for constraint in self.constraints:
            if not self._check_constraint(constraint, state):
                return False
        return True
    
    def _check_constraint(self, constraint: str, state: 'SymbolicState') -> bool:
        """
        Check a single constraint
        
        Supported DSL formats:
        - exists(field_name)
        - type(field_name) == TypeName
        - field_name != None
        - forall item in list_field: condition
        """
        constraint = constraint.strip()
        
        # exists(field_name)
        if constraint.startswith("exists(") and constraint.endswith(")"):
            field_name = constraint[7:-1].strip()
            return state.exists(field_name)
        
        # type(field_name) == TypeName
        if "type(" in constraint and "==" in constraint:
            # Simplified: check if field exists
            parts = constraint.split("==")
            if len(parts) == 2:
                type_part = parts[0].strip()
                if type_part.startswith("type(") and type_part.endswith(")"):
                    field_name = type_part[5:-1].strip()
                    return state.has_field(field_name)
        
        # field_name != None
        if "!=" in constraint and "None" in constraint:
            field_name = constraint.split("!=")[0].strip()
            return state.exists(field_name)
        
        # Default: if constraint cannot be parsed, return True (lenient strategy)
        return True


@dataclass
class Postcondition:
    """
    Postcondition Q_t
    
    Defines constraints that results and state must satisfy after calling the tool
    """
    # Result constraints: check the structure and type of return result r_t
    result_constraints: List[str] = field(default_factory=list)
    
    # State update rules: define how to update state from r_t
    state_update_rules: Dict[str, str] = field(default_factory=dict)
    # Format: {"field_name": "result.path.to.field"}
    
    # Semantic constraints: check consistency of updated state
    semantic_constraints: List[str] = field(default_factory=list)
    
    # Optional verification function
    verify_function: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None
    
    def verify_result(self, result: Dict[str, Any]) -> bool:
        """
        Verify the structure and type of tool return result
        
        Args:
            result: Raw result returned by tool
        
        Returns:
            True if result is valid, False otherwise
        """
        for constraint in self.result_constraints:
            if not self._check_result_constraint(constraint, result):
                return False
        return True
    
    def _check_result_constraint(self, constraint: str, result: Dict[str, Any]) -> bool:
        """Check result constraint"""
        constraint = constraint.strip()
        
        # Check required field exists
        if constraint.startswith("has_field(") and constraint.endswith(")"):
            field_name = constraint[10:-1].strip().strip('"\'')
            return field_name in result
        
        # Check type
        if "type(" in constraint and "==" in constraint:
            # Simplified handling
            return True
        
        # Default pass
        return True
    
    def compute_state_update(self, old_state: 'SymbolicState', 
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute state update according to update rules
        
        Args:
            old_state: Old state
            result: Tool return result
        
        Returns:
            State update dictionary
        """
        updates = {}
        
        for field_name, rule in self.state_update_rules.items():
            # Rule format: "result.path.to.field" or directly "result"
            if rule.startswith("result."):
                path = rule[7:].split(".")
                value = result
                try:
                    for key in path:
                        if isinstance(value, dict):
                            value = value[key]
                        elif isinstance(value, list) and key.isdigit():
                            value = value[int(key)]
                        else:
                            value = None
                            break
                    if value is not None:
                        updates[field_name] = value
                except (KeyError, IndexError, TypeError):
                    pass
            elif rule == "result":
                # Directly use the entire result
                updates[field_name] = result
        
        return updates
    
    def verify_semantic_constraints(self, old_state: 'SymbolicState',
                                   new_state: 'SymbolicState',
                                   result: Dict[str, Any]) -> bool:
        """
        Verify semantic constraints (consistency check)
        
        Args:
            old_state: Old state
            new_state: New state
            result: Tool return result
        
        Returns:
            True if semantic constraints are satisfied, False otherwise
        """
        if self.verify_function:
            try:
                return self.verify_function(
                    old_state.to_dict(),
                    new_state.to_dict(),
                    result
                )
            except Exception:
                return False
        
        # Parse semantic constraints
        for constraint in self.semantic_constraints:
            if not self._check_semantic_constraint(constraint, old_state, new_state):
                return False
        
        return True
    
    def _check_semantic_constraint(self, constraint: str,
                                  old_state: 'SymbolicState',
                                  new_state: 'SymbolicState') -> bool:
        """Check semantic constraint"""
        # Example: "forall s in showtimes: s.city == city"
        # Simplified here, can be more complex in practice
        return True


@dataclass
class ToolContract:
    """
    Hoare Contract Contract(t) = (P_t, Q_t)
    
    Complete definition of tool calling constraints
    """
    tool_name: str
    tool_description: str
    precondition: Precondition
    postcondition: Postcondition
    
    # Other metadata for the tool
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        return {
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "precondition": {
                "constraints": self.precondition.constraints
            },
            "postcondition": {
                "result_constraints": self.postcondition.result_constraints,
                "state_update_rules": self.postcondition.state_update_rules,
                "semantic_constraints": self.postcondition.semantic_constraints
            },
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolContract':
        """Create contract from dictionary"""
        return cls(
            tool_name=data["tool_name"],
            tool_description=data["tool_description"],
            precondition=Precondition(
                constraints=data.get("precondition", {}).get("constraints", [])
            ),
            postcondition=Postcondition(
                result_constraints=data.get("postcondition", {}).get("result_constraints", []),
                state_update_rules=data.get("postcondition", {}).get("state_update_rules", {}),
                semantic_constraints=data.get("postcondition", {}).get("semantic_constraints", [])
            ),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema")
        )

