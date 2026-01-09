"""
Precondition Checker

Checks whether state S satisfies tool precondition P_t
"""

from typing import Dict, Any, Optional, Tuple
from ..core.symbolic_state import SymbolicState
from ..contracts.contract import ToolContract, Precondition


class PreconditionChecker:
    """
    Precondition Checker
    
    Implements checking logic for S ⊨ P_t
    """
    
    def __init__(self, use_llm_verifier: bool = False, llm_client=None):
        """
        Initialize checker
        
        Args:
            use_llm_verifier: Whether to use LLM for complex predicate verification
            llm_client: LLM client (optional)
        """
        self.use_llm_verifier = use_llm_verifier
        self.llm_client = llm_client
    
    def check(self, state: SymbolicState, contract: ToolContract) -> bool:
        """
        Check if state satisfies tool precondition
        
        Args:
            state: Current symbolic state
            contract: Tool's Hoare contract
        
        Returns:
            True if satisfied, False otherwise
        """
        precondition = contract.precondition
        
        # Use contract's precondition check
        return precondition.check(state)
    
    def check_with_reason(self, state: SymbolicState, 
                          contract: ToolContract) -> Tuple[bool, str]:
        """
        Check precondition and return reason
        
        Returns:
            (whether satisfied, reason description)
        """
        satisfied = self.check(state, contract)
        
        if satisfied:
            reason = f"State satisfies precondition for tool {contract.tool_name}"
        else:
            missing = []
            for constraint in contract.precondition.constraints:
                if constraint.startswith("exists("):
                    field_name = constraint[7:-1].strip()
                    if not state.exists(field_name):
                        missing.append(field_name)
            
            if missing:
                reason = f"Missing required fields: {', '.join(missing)}"
            else:
                reason = f"State does not satisfy precondition for tool {contract.tool_name}"
        
        return satisfied, reason

