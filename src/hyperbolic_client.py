"""
Hyperbolic API clients with async support.
"""
import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
import httpx
import aiofiles
from typing import Dict, List, Any
import numpy as np

class HyperbolicBaseClient:
    """
    Async client for Hyperbolic base models (completions API).
    Supports logprobs extraction for True/False classification.
    """
    
    def __init__(
        self,
        log_calls: bool = False,
        log_dir: str | None = None,
        graceful_failure: bool = False,
        call_type: str = "default"
    ):
        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        assert self.api_key, "HYPERBOLIC_API_KEY environment variable not set"
        
        self.base_url = "https://api.hyperbolic.xyz/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Increased timeout for parallel requests and network instability
        self._client = httpx.AsyncClient(timeout=180.0)
        
        # Logging configuration
        self.log_calls = log_calls
        self.log_dir = log_dir
        self.graceful_failure = graceful_failure
        self.call_type = call_type
    
        print(f"Setting up client with log_calls={self.log_calls}, log_dir={self.log_dir}, call_type={self.call_type}")
        
        # Create log directory if logging is enabled
        if self.log_calls and self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _log_call(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        duration: float
    ):
        """Log API call to JSONL file."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": request_data.get("model"),
            "call_type": self.call_type,
            "request": request_data,
            "response": response_data,
            "duration": duration
        }
        
        log_file = Path(self.log_dir) / f"{self.call_type}.jsonl"
        print(f"Logging call to {log_file}")
        async with aiofiles.open(log_file, mode='a') as f:
            await f.write(json.dumps(log_entry) + '\n')
    
    async def get_label_logprobs(
        self, 
        prompt: str,
        model: str,
        labels: List[str] = ["True", "False"],
        max_retries: int = 5,
        base_retry_delay: float = 1.0
    ) -> Dict[str, float]:
        """
        Get log probabilities for specified labels with retry logic for rate limiting.
        
        Args:
            prompt: The prompt string (should end before the label token)
            model: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B")
            labels: List of label strings to extract logprobs for
            max_retries: Maximum number of retry attempts for rate limiting
            base_retry_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Dict mapping label -> log probability
            
        Raises:
            ValueError: If label not found in top logprobs
            httpx.HTTPStatusError: If non-rate-limit errors occur
        """
        start_time = time.time()
        
        data = {
            "prompt": prompt,
            "model": model,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5
        }
        
        # Retry loop with exponential backoff for rate limiting
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                break  # Success, exit retry loop
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 429:  # Rate limit error
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        delay = base_retry_delay * (2 ** attempt)
                        print(f"Rate limited (429). Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f"Rate limit error persisted after {max_retries} attempts")
                        raise
                else:
                    # Non-rate-limit HTTP error, raise immediately
                    raise
            except (httpx.ReadTimeout, httpx.ReadError, httpx.ConnectError) as e:
                # Network errors - retry with backoff
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_retry_delay * (2 ** attempt)
                    print(f"Network error: {type(e).__name__}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"Network error persisted after {max_retries} attempts")
                    raise
        
        if last_exception and 'result' not in locals():
            # All retries exhausted
            raise last_exception
        
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
                # TODO: This is a hack. More elegant handling to check that logprobs account for most of mass would be good!
                if self.graceful_failure:
                    print(f"LENNIE WARNING! Label '{label}' not found in top logprobs. Returning 0.0 for graceful failure.")
                    label_logprobs[label] = -np.inf
                else:
                    raise ValueError(
                        f"Label '{label}' not found in top logprobs. "
                        f"Available tokens: {list(top_logprobs.keys())}"
                    )
        
        duration = time.time() - start_time
        
        # Log the call
        if self.log_calls:
            await self._log_call(
                request_data=data,
                response_data={
                    "completion": result['choices'][0]['text'],
                    "logprobs": label_logprobs,
                    "top_logprobs": top_logprobs
                },
                duration=duration
            )
        
        return label_logprobs


class HyperbolicChatClient:
    """
    Async client for Hyperbolic chat models (chat completions API).
    Does not support logprobs.
    """
    
    def __init__(
        self,
        log_calls: bool = False,
        log_dir: str | None = None,
        graceful_failure: bool = False,
        call_type: str = "default"
    ):
        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        assert self.api_key, "HYPERBOLIC_API_KEY environment variable not set"
        
        self.base_url = "https://api.hyperbolic.xyz/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Increased timeout for parallel requests and network instability
        self._client = httpx.AsyncClient(timeout=180.0)
        
        # Logging configuration
        self.log_calls = log_calls
        self.log_dir = log_dir
        self.graceful_failure = graceful_failure
        self.call_type = call_type
        
        # Create log directory if logging is enabled
        if self.log_calls and self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _log_call(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        duration: float
    ):
        """Log API call to JSONL file."""
        if not self.log_calls or not self.log_dir:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": request_data.get("model"),
            "call_type": self.call_type,
            "request": request_data,
            "response": response_data,
            "duration": duration
        }
        
        log_file = Path(self.log_dir) / f"{self.call_type}.jsonl"
        async with aiofiles.open(log_file, mode='a') as f:
            await f.write(json.dumps(log_entry) + '\n')
    
    async def get_chat_prediction(
        self, 
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """
        Get prediction from chat model with retry logic for empty responses.
        
        Args:
            prompt: User prompt
            model: Chat model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B-Instruct")
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts for empty responses
            
        Returns:
            Model's text response (empty string if all retries fail and graceful_failure=True)
            
        Raises:
            RuntimeError: If all retries fail and graceful_failure=False
        """
        start_time = time.time()
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "max_tokens": 10,
            "temperature": temperature,
            "top_p": 0.9, # needed for sampling at temperature>0 to make evaluate!
        }
        
        completion = ""
        last_result = None
        
        for attempt in range(1, max_retries + 1):
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            last_result = result
            
            # Extract content, handling None case
            content = result['choices'][0]['message'].get('content')
            if content is None:
                completion = ""
            else:
                completion = content.strip()
            
            # Check if we got a valid response
            if completion:
                # Success! Log and return
                duration = time.time() - start_time
                await self._log_call(
                    request_data=data,
                    response_data={
                        "completion": completion,
                        "finish_reason": result['choices'][0].get('finish_reason'),
                        "attempt": attempt
                    },
                    duration=duration
                )
                return completion
            
            # Empty response - warn and retry
            print(f"WARNING: Empty response on attempt {attempt}/{max_retries}")
        
        # All retries exhausted
        duration = time.time() - start_time
        
        # Log the final failed attempt
        if last_result:
            await self._log_call(
                request_data=data,
                response_data={
                    "completion": "",
                    "finish_reason": last_result['choices'][0].get('finish_reason'),
                    "attempt": max_retries,
                    "all_retries_failed": True
                },
                duration=duration
            )
        
        if self.graceful_failure:
            print(f"WARNING: All {max_retries} retries failed. Returning empty string (graceful_failure=True)")
            return ""
        else:
            raise RuntimeError(
                f"Failed to get non-empty response after {max_retries} attempts. "
                f"Last finish_reason: {last_result['choices'][0].get('finish_reason') if last_result else 'unknown'}"
            )
