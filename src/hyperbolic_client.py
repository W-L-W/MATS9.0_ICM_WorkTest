"""
Simple Hyperbolic API client for ICM algorithm.
Supports completions API with logprobs extraction.
"""
import os
import requests
from typing import Dict, List, Any


class HyperbolicClient:
    """Client for Hyperbolic completions API."""
    
    def __init__(self):
        """
        Initialize Hyperbolic client.
        
        Args:
            api_key: Hyperbolic API key
        """
        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        self.base_url = "https://api.hyperbolic.xyz/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _get_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a request to Hyperbolic completions API.
        
        Args:
            prompt: The prompt string
            model: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            logprobs: Number of top logprobs to return
            **kwargs: Additional API parameters (top_p, etc.)
            
        Returns:
            Full API response as dict
            
        Raises:
            requests.HTTPError: If API request fails
        """
        data = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": logprobs,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_label_logprobs(
        self, 
        prompt: str,
        model: str,
        labels: List[str] = ["True", "False"],
        raise_failures: bool = True
    ) -> Dict[str, float]:
        """
        Get log probabilities for specified labels.
        
        Args:
            prompt: The prompt string (should end before the label token)
            model: Model identifier
            labels: List of label strings to extract logprobs for
            raise_failures: If True, raise error when label not in top logprobs
            
        Returns:
            Dict mapping label -> log probability
            
        Raises:
            ValueError: If raise_failures=True and label not found in top logprobs
            
        Example:
            >>> client = HyperbolicClient(api_key="...")
            >>> prompt = "Question: X\\nClaim: Y\\nI think this Claim is"
            >>> logprobs = client.get_label_logprobs(prompt, "meta-llama/Meta-Llama-3.1-405B")
            >>> # Returns: {"True": -0.5, "False": -2.3}
        """
        result = self._get_response(
            prompt=prompt,
            model=model,
            max_tokens=1,
            temperature=0.0,
            logprobs=5
        )
        
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
                # Label not in top logprobs
                if raise_failures:
                    raise ValueError(
                        f"Label '{label}' not found in top logprobs. "
                        f"Available tokens: {list(top_logprobs.keys())}"
                    )
                else:
                    # Assign very negative value as fallback
                    label_logprobs[label] = -100.0
        
        return label_logprobs
