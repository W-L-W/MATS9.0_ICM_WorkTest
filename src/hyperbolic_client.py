"""
Hyperbolic API clients with async support.
"""
import os
import httpx
from typing import Dict, List


class HyperbolicBaseClient:
    """
    Async client for Hyperbolic base models (completions API).
    Supports logprobs extraction for True/False classification.
    """
    
    def __init__(self):
        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        assert self.api_key, "HYPERBOLIC_API_KEY environment variable not set"
        
        self.base_url = "https://api.hyperbolic.xyz/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self._client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get_label_logprobs(
        self, 
        prompt: str,
        model: str,
        labels: List[str] = ["True", "False"]
    ) -> Dict[str, float]:
        """
        Get log probabilities for specified labels.
        
        Args:
            prompt: The prompt string (should end before the label token)
            model: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B")
            labels: List of label strings to extract logprobs for
            
        Returns:
            Dict mapping label -> log probability
            
        Raises:
            ValueError: If label not found in top logprobs
        """
        data = {
            "prompt": prompt,
            "model": model,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5
        }
        
        response = await self._client.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract top_logprobs for first token
        top_logprobs = result['choices'][0]['logprobs']['top_logprobs'][0]
        
        # Extract logprobs for our labels
        label_logprobs = {}
        for label in labels:
            # Try exact match first, then with leading space
            if label in top_logprobs:
                label_logprobs[label] = top_logprobs[label]
            elif f" {label}" in top_logprobs:
                label_logprobs[label] = top_logprobs[f" {label}"]
            else:
                raise ValueError(
                    f"Label '{label}' not found in top logprobs. "
                    f"Available tokens: {list(top_logprobs.keys())}"
                )
        
        return label_logprobs


class HyperbolicChatClient:
    """
    Async client for Hyperbolic chat models (chat completions API).
    Does not support logprobs.
    """
    
    def __init__(self):
        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        assert self.api_key, "HYPERBOLIC_API_KEY environment variable not set"
        
        self.base_url = "https://api.hyperbolic.xyz/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self._client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get_chat_prediction(
        self, 
        prompt: str,
        model: str,
        temperature: float = 0.7
    ) -> str:
        """
        Get prediction from chat model.
        
        Args:
            prompt: User prompt
            model: Chat model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B-Instruct")
            temperature: Sampling temperature
            
        Returns:
            Model's text response
        """
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "max_tokens": 10,
            "temperature": temperature
        }
        
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content'].strip()
