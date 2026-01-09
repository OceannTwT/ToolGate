"""
Tool Retriever

Implements RAG + rerank tool retrieval pipeline
Supports BM25 and Embedding vector retrieval
"""

from typing import List, Dict, Any, Optional
import json
import os
from collections import defaultdict
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import cohere
except ImportError:
    cohere = None

try:
    import torch
    from torch.nn.functional import cosine_similarity
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    cosine_similarity = None

try:
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoTokenizer = None

from ..contracts.contract import ToolContract


class ToolRetriever:
    """
    Tool Retriever
    
    Implements:
    1. BM25 retrieval (word-based matching)
    2. Embedding vector retrieval (semantic similarity using qwen3-embedding)
    3. Cohere reranking
    """
    
    def __init__(self, 
                 tool_contracts: Dict[str, ToolContract],
                 embedding_model: Optional[Any] = None,
                 embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 cohere_api_key: Optional[str] = None,
                use_bm25: bool = False,  # Default: do not use BM25, use embedding
                use_embedding: bool = True,  # Default: use embedding
                 use_rerank: bool = True):
        """
        Initialize retriever
        
        Args:
            tool_contracts: Tool contract dictionary {tool_name: contract}
            embedding_model: Optional preloaded embedding model (if provided, will use it)
            embedding_model_name: Embedding model name (default: Qwen3-Embedding-0.6B)
            cohere_api_key: Cohere API key (for rerank)
            use_bm25: Whether to use BM25
            use_embedding: Whether to use Embedding vector retrieval
            use_rerank: Whether to use rerank
        """
        self.tool_contracts = tool_contracts
        self.embedding_model_name = embedding_model_name
        self.cohere_api_key = cohere_api_key
        self.use_bm25 = use_bm25 and BM25Okapi is not None
        self.use_embedding = use_embedding
        self.use_rerank = use_rerank and cohere is not None and cohere_api_key
        
        # Initialize Embedding model
        if self.use_embedding:
            if embedding_model:
                self.embedding_model = embedding_model
                self.embedding_tokenizer = None  # Assume model already contains tokenizer
            elif _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE:
                try:
                    print(f"Loading Embedding model: {embedding_model_name}")
                    self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
                    self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
                    self.embedding_model.eval()  # Set to evaluation mode
                    if torch.cuda.is_available():
                        self.embedding_model = self.embedding_model.cuda()
                    print(f"✓ Embedding model loaded successfully")
                except Exception as e:
                    print(f"Warning: Unable to load Embedding model {embedding_model_name}: {e}")
                    print("   Will fall back to BM25 or simple matching")
                    self.use_embedding = False
                    self.embedding_model = None
                    self.embedding_tokenizer = None
            else:
                print("Warning: transformers or torch not installed, cannot use Embedding retrieval")
                print("   Installation: pip install transformers torch")
                self.use_embedding = False
                self.embedding_model = None
                self.embedding_tokenizer = None
        else:
            self.embedding_model = None
            self.embedding_tokenizer = None
        
        # Build indices
        self._build_indices()
        
        # Initialize Cohere client
        if self.use_rerank:
            try:
                self.cohere_client = cohere.Client(api_key=cohere_api_key)
            except Exception:
                self.use_rerank = False
                self.cohere_client = None
        else:
            self.cohere_client = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector (numpy array)
        """
        if not self.embedding_model or not self.embedding_tokenizer:
            raise ValueError("Embedding model not initialized")
        
        # Encode text using tokenizer
        inputs = self.embedding_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512  # Limit maximum length
        )
        
        # Move to GPU if available
        device = next(self.embedding_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use [CLS] token embedding or average pooling
            if hasattr(outputs, 'last_hidden_state'):
                # Average pool all token embeddings
                embeddings = outputs.last_hidden_state
                # Use attention mask for average pooling
                if 'attention_mask' in inputs:
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                else:
                    embeddings = embeddings.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                # If no explicit output, use first token
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Convert to numpy and normalize (for cosine similarity)
        embedding = embeddings.cpu().numpy()[0]
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _build_indices(self):
        """Build retrieval indices"""
        self.tool_names = list(self.tool_contracts.keys())
        self.tool_texts = []
        
        # Build text representation for each tool (for retrieval)
        for tool_name, contract in self.tool_contracts.items():
            text_parts = []
            
            # Tool description (most important, should contain sufficient information)
            if contract.tool_description:
                text_parts.append(contract.tool_description)
            
            # Tool name (may contain functional information)
            text_parts.append(f"Tool: {contract.tool_name}")
            
            # Add input parameter information (helps understand tool functionality)
            if contract.input_schema and 'properties' in contract.input_schema:
                param_names = list(contract.input_schema['properties'].keys())
                if param_names:
                    text_parts.append(f"Parameters: {', '.join(param_names[:5])}")  # Only take first 5 parameters
            
            # Add precondition summary (helps understand tool usage scenarios)
            if contract.precondition.constraints:
                # Simplify precondition representation
                constraints_str = ', '.join(contract.precondition.constraints[:3])  # Only take first 3
                text_parts.append(f"Requires: {constraints_str}")
            
            # Add postcondition summary (helps understand tool output)
            if contract.postcondition.state_update_rules:
                updates = list(contract.postcondition.state_update_rules.keys())
                if updates:
                    text_parts.append(f"Returns: {', '.join(updates[:3])}")  # Only take first 3
            
            self.tool_texts.append(" ".join(text_parts))
        
        # Build BM25 index
        if self.use_bm25:
            tokenized_texts = [text.split() for text in self.tool_texts]
            self.bm25 = BM25Okapi(tokenized_texts)
        else:
            self.bm25 = None
        
        # Build Embedding index
        if self.use_embedding and self.embedding_model:
            print(f"Generating embedding vectors for {len(self.tool_texts)} tools...")
            self.tool_embeddings = []
            for i, text in enumerate(self.tool_texts):
                try:
                    embedding = self._get_embedding(text)
                    self.tool_embeddings.append(embedding)
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(self.tool_texts)} tools")
                except Exception as e:
                    print(f"Warning: Failed to generate embedding for tool {self.tool_names[i]}: {e}")
                    # Use zero vector as fallback
                    if self.tool_embeddings:
                        dim = len(self.tool_embeddings[0])
                    else:
                        dim = 768  # Default dimension
                    self.tool_embeddings.append(np.zeros(dim))
            print(f"✓ Embedding vectors for all tools generated")
            # Convert to numpy array for batch computation
            self.tool_embeddings = np.array(self.tool_embeddings)
        else:
            self.tool_embeddings = None
    
    def retrieve(self, query: str, state_summary: str = "", 
                top_k: int = 25) -> List[str]:
        """
        Retrieve candidate tools
        
        Args:
            query: Query text (sub-goal description)
            state_summary: State summary
            top_k: Number of candidates to return
        
        Returns:
            Sorted list of tool names
        """
        # Combine query (prioritize query, state_summary as supplement)
        # If query is empty or too short, use state_summary
        if not query or len(query.strip()) < 3:
            full_query = state_summary.strip()
        else:
            full_query = query.strip()
            if state_summary:
                full_query = f"{query} {state_summary}".strip()
        
        # Embedding vector retrieval (preferred, more accurate)
        if self.use_embedding and self.tool_embeddings is not None:
            try:
                # Generate embedding for query
                query_embedding = self._get_embedding(full_query)
                
                # Calculate cosine similarity with all tools
                # Since already L2 normalized, cosine similarity = dot product
                similarities = np.dot(self.tool_embeddings, query_embedding)
                
                # Get top_k
                top_indices = np.argsort(similarities)[::-1][:top_k]
                candidate_tools = [self.tool_names[i] for i in top_indices]
                
                # Optional: print similarity scores (for debugging)
                # print(f"Top {min(5, len(candidate_tools))} tool similarities:")
                # for i, tool_name in enumerate(candidate_tools[:5]):
                #     idx = self.tool_names.index(tool_name)
                #     print(f"  {tool_name}: {similarities[idx]:.4f}")
                
            except Exception as e:
                print(f"Warning: Embedding retrieval failed: {e}")
                # Fall back to BM25 or simple matching
                candidate_tools = self._fallback_retrieve(full_query, top_k)
        
        # BM25 retrieval (as alternative or combination)
        elif self.use_bm25:
            candidate_tools = self._bm25_retrieve(full_query, top_k)
        
        # Simple text matching (fallback)
        else:
            candidate_tools = self._simple_match(full_query, top_k)
        
        # Rerank (use original query, not full_query)
        if self.use_rerank and len(candidate_tools) > 1:
            rerank_query = query if query else state_summary
            candidate_tools = self._rerank(rerank_query, candidate_tools)
        
        return candidate_tools[:top_k]
    
    def _bm25_retrieve(self, query: str, top_k: int) -> List[str]:
        """BM25 retrieval"""
        query_tokens = query.lower().split()
        # Filter out words that are too short (may be stop words or noise)
        query_tokens = [t for t in query_tokens if len(t) > 2]
        
        if query_tokens:
            scores = self.bm25.get_scores(query_tokens)
            # Get top_k
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            return [self.tool_names[i] for i in top_indices]
        else:
            # If query tokens are too short, use simple matching
            return self._simple_match(query, top_k)
    
    def _fallback_retrieve(self, query: str, top_k: int) -> List[str]:
        """Fallback retrieval method"""
        if self.use_bm25:
            return self._bm25_retrieve(query, top_k)
        else:
            return self._simple_match(query, top_k)
    
    def _simple_match(self, query: str, top_k: int) -> List[str]:
        """Simple text matching (fallback)"""
        query_lower = query.lower()
        scores = []
        
        for i, text in enumerate(self.tool_texts):
            text_lower = text.lower()
            # Simple keyword matching
            score = sum(1 for word in query_lower.split() if word in text_lower)
            scores.append((score, i))
        
        scores.sort(reverse=True)
        return [self.tool_names[i] for _, i in scores[:top_k]]
    
    def _rerank(self, query: str, candidate_tools: List[str]) -> List[str]:
        """
        Use Cohere rerank
        
        Args:
            query: Query text
            candidate_tools: List of candidate tools
        
        Returns:
            Re-ranked list of tools
        """
        if not self.cohere_client:
            return candidate_tools
        
        try:
            # Build document list
            documents = [
                self.tool_texts[self.tool_names.index(tool_name)]
                for tool_name in candidate_tools
            ]
            
            # Call rerank API
            results = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=len(candidate_tools)
            )
            
            # Re-order according to rerank results
            reranked = [
                candidate_tools[result.index]
                for result in results.results
            ]
            
            return reranked
        except Exception as e:
            # If rerank fails, return original order
            print(f"Rerank failed: {e}")
            return candidate_tools

