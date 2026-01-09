"""
Postcondition Verifier

Verifies whether tool call results satisfy postcondition Q_t
"""

from typing import Dict, Any, Optional, Tuple, Union
from ..core.symbolic_state import SymbolicState
from ..contracts.contract import ToolContract, Postcondition


class PostconditionVerifier:
    """
    Postcondition Verifier
    
    Implements verification logic for Verify(S_k, r_t, Q_t)
    """
    
    def __init__(self, use_llm_verifier: bool = False, llm_client=None):
        """
        Initialize verifier
        
        Args:
            use_llm_verifier: Whether to use LLM for complex semantic verification
            llm_client: LLM client (optional)
        """
        self.use_llm_verifier = use_llm_verifier
        self.llm_client = llm_client
    
    def verify(self, old_state: SymbolicState,
              contract: ToolContract,
              result: Dict[str, Any]) -> Tuple[bool, Union[str, None], Dict[str, Any]]:
        """
        Verify tool call result
        
        Args:
            old_state: State before call
            contract: Tool's Hoare contract
            result: Raw result returned by tool
        
        Returns:
            (whether verification passed, error message, state update dictionary)
        """
        postcondition = contract.postcondition
        
        # 1. Verify result structure
        if not postcondition.verify_result(result):
            return False, "Tool return result does not satisfy structural constraints", {}
        
        # 2. Compute state update
        updates = postcondition.compute_state_update(old_state, result)
        
        # 3. Construct new state
        new_state = old_state.update(updates)
        
        # 4. Verify semantic constraints
        if not postcondition.verify_semantic_constraints(old_state, new_state, result):
            return False, "State after update does not satisfy semantic constraints", {}
        
        # 5. If all pass, return success
        return True, None, updates
    
    def verify_with_details(self, old_state: SymbolicState,
                           contract: ToolContract,
                           result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detailed verification and return detailed information
        
        Returns:
            Dictionary containing verification results
        """
        success, error_msg, updates = self.verify(old_state, contract, result)
        
        return {
            "success": success,
            "error_message": error_msg,
            "updates": updates,
            "result_valid": contract.postcondition.verify_result(result) if not success else True
        }

