"""
State Manager

Responsible for state initialization and updates
"""

from typing import Dict, Any, Optional, List
from ..core.symbolic_state import SymbolicState, StateType


class StateManager:
    """
    State Manager
    
    Responsibilities:
    1. Initialize state from user query and history
    2. Manage state updates
    """
    
    def __init__(self):
        """Initialize state manager"""
        pass
    
    def initialize_state(self, query: str, history: Optional[List[Dict[str, Any]]] = None,
                        extracted_entities: Optional[Dict[str, Any]] = None) -> SymbolicState:
        """
        Initialize state S_0
        
        Args:
            query: User query
            history: Conversation history
            extracted_entities: Entities extracted from query (optional)
        
        Returns:
            Initialized symbolic state
        """
        state = SymbolicState()
        
        # If extracted entities exist, use them directly
        if extracted_entities:
            for name, value_info in extracted_entities.items():
                if isinstance(value_info, tuple):
                    value, type_str = value_info
                    state_type = self._parse_type(type_str)
                else:
                    value = value_info
                    state_type = StateType.STRING
                
                state.set_field(name, value, state_type)
        
        # Otherwise, try to extract from query (simplified)
        # In practice, can use NER or LLM extraction
        
        return state
    
    def _parse_type(self, type_str: str) -> StateType:
        """Parse type string"""
        type_str = type_str.strip()
        if type_str.endswith('?'):
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
        
        try:
            return StateType[type_str.upper()]
        except KeyError:
            pass
        
        return type_mapping.get(type_str.lower(), StateType.STRING)
    
    def update_state(self, old_state: SymbolicState, 
                    updates: Dict[str, Any]) -> SymbolicState:
        """
        Update state
        
        Args:
            old_state: Old state
            updates: Update dictionary
        
        Returns:
            New state
        """
        return old_state.update(updates)

