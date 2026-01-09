"""
Contract Generator

Automatically generates Hoare contracts (P_t, Q_t) from tool documentation
"""

from typing import Dict, Any, List, Optional
import json
from .contract import ToolContract, Precondition, Postcondition


class ContractGenerator:
    """
    Contract Generator
    
    Automatically generates contracts from tool natural language descriptions and schemas
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize generator
        
        Args:
            llm_client: Optional LLM client (for generating contracts)
        """
        self.llm_client = llm_client
    
    def generate_contract(self, tool_info: Dict[str, Any]) -> ToolContract:
        """
        Generate contract for tool
        
        Args:
            tool_info: Tool information containing:
                - tool_name: Tool name
                - tool_description: Tool description
                - input_schema: Input parameter schema
                - output_schema: Output result schema
        
        Returns:
            ToolContract object
        """
        tool_name = tool_info.get("tool_name", "")
        description = tool_info.get("tool_description", "")
        input_schema = tool_info.get("input_schema", {})
        output_schema = tool_info.get("output_schema", {})
        
        # Generate precondition
        precondition = self._generate_precondition(
            tool_name, description, input_schema
        )
        
        # Generate postcondition
        postcondition = self._generate_postcondition(
            tool_name, description, output_schema
        )
        
        return ToolContract(
            tool_name=tool_name,
            tool_description=description,
            precondition=precondition,
            postcondition=postcondition,
            input_schema=input_schema,
            output_schema=output_schema
        )
    
    def _generate_precondition(self, tool_name: str, description: str,
                              input_schema: Dict[str, Any]) -> Precondition:
        """
        Generate precondition
        
        Extract required parameters from input schema and generate exists() constraints
        """
        constraints = []
        
        # Extract required parameters from input_schema
        if isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            for param_name in required:
                constraints.append(f"exists({param_name})")
        
        # Can also extract from description (using LLM or rules)
        # Simplified here
        
        return Precondition(constraints=constraints)
    
    def _generate_postcondition(self, tool_name: str, description: str,
                               output_schema: Dict[str, Any]) -> Postcondition:
        """
        Generate postcondition
        
        Extract fields from output schema and generate result constraints and state update rules
        """
        result_constraints = []
        state_update_rules = {}
        
        # Extract fields from output_schema
        if isinstance(output_schema, dict):
            properties = output_schema.get("properties", {})
            
            for field_name, field_schema in properties.items():
                # Check required fields
                result_constraints.append(f'has_field("{field_name}")')
                
                # Generate state update rules (assume tool return field names directly correspond to state field names)
                state_update_rules[field_name] = f"result.{field_name}"
        
        # Can also infer from description (using LLM)
        # Example: if tool is "SearchMovieShowtimes", may update "showtimes" field
        
        return Postcondition(
            result_constraints=result_constraints,
            state_update_rules=state_update_rules
        )
    
    def generate_from_tool_json(self, tool_json: Dict[str, Any]) -> ToolContract:
        """
        Generate contract from Toolbench format tool JSON
        
        Args:
            tool_json: Tool definition in Toolbench format
        
        Returns:
            ToolContract object
        """
        # Parse Toolbench format
        tool_name = tool_json.get("tool_name", "")
        description = tool_json.get("tool_description", "")
        
        # Extract input/output schemas
        input_schema = {}
        output_schema = {}
        
        # Extract from API definition
        if "api_list" in tool_json:
            # Handle multiple APIs
            api = tool_json["api_list"][0] if tool_json["api_list"] else {}
            input_schema = api.get("parameters", {})
            output_schema = api.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {}).get("schema", {})
        
        return self.generate_contract({
            "tool_name": tool_name,
            "tool_description": description,
            "input_schema": input_schema,
            "output_schema": output_schema
        })

