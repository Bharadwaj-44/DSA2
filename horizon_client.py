"""
Horizon API Client for Elevance Health - Production Version
Replaces snowflake_cortex_client.py

Features:
- Token refresh logic (tokens expire every 2 hours)
- Automatic retry with exponential backoff
- Conversation history management with token limits
- Schema validation and error handling
- Debug logging with secret masking

Reference: Horizon API Documentation
- Endpoint: /v2/text/chats
- Authentication: Bearer token (expires every 2 hours)
- Format: Stateless, send full conversation history
"""

import requests
import json
import time
import random
from typing import List, Dict, Iterator, Optional, Callable
from datetime import datetime, timedelta
import tiktoken
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
#                         CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_BASE_URL = "https://api.horizon.elevancehealth.com"
TOKEN_EXPIRY_SECONDS = 7200  # 2 hours
TOKEN_REFRESH_BUFFER = 300   # Refresh 5 minutes before expiry
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0    # seconds
MAX_RETRY_DELAY = 10.0       # seconds
DEFAULT_TIMEOUT = 120        # seconds
DEFAULT_MAX_TOKENS = 7000    # Maximum tokens in conversation history


# ============================================================================
#                         RESPONSE CLASSES
# ============================================================================

class UsageStats:
    """Token usage statistics"""
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class Choice:
    """Response choice"""
    def __init__(self, message: Dict[str, str], finish_reason: str = "stop"):
        self.message = type('Message', (), message)()
        self.finish_reason = finish_reason
        self.delta = type('Delta', (), {'content': message.get('content', '')})()


class CompletionResponse:
    """Response from completion API"""
    def __init__(self, content: str, usage: UsageStats):
        self.choices = [Choice({"role": "assistant", "content": content})]
        self.usage = usage


class StreamingChunk:
    """Streaming chunk response"""
    def __init__(self, content: str):
        delta = type('Delta', (), {'content': content})()
        choice = type('Choice', (), {'delta': delta})()
        self.choices = [choice]


# ============================================================================
#                         EXCEPTION CLASSES
# ============================================================================

class HorizonAPIError(Exception):
    """Base exception for Horizon API errors"""
    pass


class TokenExpiredError(HorizonAPIError):
    """Token has expired and needs refresh"""
    pass


class RateLimitError(HorizonAPIError):
    """Rate limit exceeded"""
    pass


class HorizonServerError(HorizonAPIError):
    """Server error (5xx)"""
    pass


# ============================================================================
#                         UTILITY FUNCTIONS
# ============================================================================

def mask_secret(secret: str, show_chars: int = 4) -> str:
    """
    Mask secrets for logging
    
    Args:
        secret: Secret string to mask
        show_chars: Number of characters to show at start/end
    
    Returns:
        Masked string like "eyJ...w5c"
    """
    if not secret or len(secret) <= show_chars * 2:
        return "***"
    return f"{secret[:show_chars]}...{secret[-show_chars:]}"


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken
    
    Args:
        text: Text to estimate
    
    Returns:
        Approximate token count
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def calculate_retry_delay(attempt: int, base_delay: float = INITIAL_RETRY_DELAY) -> float:
    """
    Calculate exponential backoff with jitter
    
    Args:
        attempt: Retry attempt number (0-indexed)
        base_delay: Base delay in seconds
    
    Returns:
        Delay in seconds with jitter
    """
    # Exponential backoff: base_delay * 2^attempt
    delay = min(base_delay * (2 ** attempt), MAX_RETRY_DELAY)
    
    # Add jitter (±25%)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    
    return delay + jitter


# ============================================================================
#                         CONVERSATION HISTORY MANAGER
# ============================================================================

class ConversationBuffer:
    """
    Manages conversation history with token limits
    
    Features:
    - Tracks token count
    - Automatically trims old messages when limit exceeded
    - Preserves system message and recent context
    """
    
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = []
        self.system_message: Optional[Dict[str, str]] = None
    
    def add_message(self, role: str, content: str):
        """Add a message to the buffer"""
        message = {"role": role, "content": content}
        
        if role == "system":
            self.system_message = message
        else:
            self.messages.append(message)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get messages for API request, trimmed to token limit
        
        Returns:
            List of messages with system message first
        """
        # Start with system message
        result = []
        if self.system_message:
            result.append(self.system_message)
        
        # Calculate tokens
        total_tokens = estimate_tokens(self.system_message["content"]) if self.system_message else 0
        
        # Add messages from most recent, working backwards
        trimmed_messages = []
        for msg in reversed(self.messages):
            msg_tokens = estimate_tokens(msg["content"])
            
            if total_tokens + msg_tokens > self.max_tokens:
                # Token limit reached
                print(f"DEBUG: Token limit reached. Trimmed {len(self.messages) - len(trimmed_messages)} old messages")
                break
            
            trimmed_messages.insert(0, msg)
            total_tokens += msg_tokens
        
        result.extend(trimmed_messages)
        
        print(f"DEBUG: Conversation buffer: {len(result)} messages, ~{total_tokens} tokens")
        
        return result
    
    def clear(self):
        """Clear all messages except system message"""
        self.messages = []


# ============================================================================
#                         HORIZON CLIENT
# ============================================================================

class HorizonClient:
    """
    Production-ready Horizon API Client
    
    Features:
    - Automatic token refresh (tokens expire every 2 hours)
    - Retry logic with exponential backoff
    - Conversation history management
    - Schema validation
    - Secret masking in logs
    
    Reference: Horizon API Documentation
    """
    
    def __init__(self, 
                 config_or_token,
                 base_url: str = None,
                 token_provider: Optional[Callable[[], str]] = None,
                 max_context_tokens: int = DEFAULT_MAX_TOKENS):
        """
        Initialize Horizon client
        
        Args:
            config_or_token: Config dict/object OR token string
            base_url: Base URL for Horizon API
            token_provider: Optional callback to refresh token (returns new token)
            max_context_tokens: Max tokens for conversation history
        """
        # Parse configuration
        self._parse_config(config_or_token, base_url)
        
        # Token management
        self.token_provider = token_provider
        self.token_issued_at = datetime.now()
        
        # Conversation management
        self.max_context_tokens = max_context_tokens
        
        # Debug output
        print(f"DEBUG: Horizon client initialized")
        print(f"DEBUG: Base URL: {self.base_url}")
        print(f"DEBUG: Token: {mask_secret(self.token)}")
        print(f"DEBUG: Max context tokens: {self.max_context_tokens}")
        
        # Create chat interface
        self.chat = self.ChatCompletion(self)
    
    def _parse_config(self, config_or_token, base_url):
        """Parse configuration from various input formats"""
        if isinstance(config_or_token, dict):
            # Config dict
            config = config_or_token
            
            if 'horizon' in config:
                horizon_config = config['horizon']
                self.token = horizon_config.get('token')
                self.base_url = horizon_config.get('base_url', DEFAULT_BASE_URL)
                self.max_context_tokens = horizon_config.get('max_context_tokens', DEFAULT_MAX_TOKENS)
            else:
                self.token = config.get('token') or config.get('api_key')
                self.base_url = config.get('base_url', DEFAULT_BASE_URL)
                self.max_context_tokens = config.get('max_context_tokens', DEFAULT_MAX_TOKENS)
                
        elif hasattr(config_or_token, 'horizon') or hasattr(config_or_token, 'token'):
            # Config object
            config = config_or_token
            
            if hasattr(config, 'horizon'):
                horizon_config = config.horizon
                self.token = getattr(horizon_config, 'token', None)
                self.base_url = getattr(horizon_config, 'base_url', DEFAULT_BASE_URL)
                self.max_context_tokens = getattr(horizon_config, 'max_context_tokens', DEFAULT_MAX_TOKENS)
            else:
                self.token = getattr(config, 'token', None) or getattr(config, 'api_key', None)
                self.base_url = getattr(config, 'base_url', DEFAULT_BASE_URL)
                self.max_context_tokens = getattr(config, 'max_context_tokens', DEFAULT_MAX_TOKENS)
        else:
            # Token string
            self.token = config_or_token
            self.base_url = base_url if base_url else DEFAULT_BASE_URL
            self.max_context_tokens = DEFAULT_MAX_TOKENS
        
        # Clean base URL
        self.base_url = self.base_url.rstrip('/')
        
        if not self.token:
            raise ValueError("Token is required. Please provide a valid Horizon API token.")
    
    def _should_refresh_token(self) -> bool:
        """
        Check if token should be refreshed proactively
        
        Reference: Migration notes - "refresh proactively at 105 minutes"
        
        Returns:
            True if token should be refreshed
        """
        age = (datetime.now() - self.token_issued_at).total_seconds()
        refresh_threshold = TOKEN_EXPIRY_SECONDS - TOKEN_REFRESH_BUFFER
        
        if age >= refresh_threshold:
            print(f"DEBUG: Token age {age:.0f}s >= {refresh_threshold}s, should refresh")
            return True
        
        return False
    
    def _refresh_token(self):
        """
        Refresh the authentication token
        
        Reference: "Authentication" collection - tokens expire every 2 hours
        """
        if not self.token_provider:
            print("WARNING: Token may be expired but no token_provider configured")
            return
        
        try:
            print("DEBUG: Refreshing token...")
            new_token = self.token_provider()
            
            if new_token:
                self.token = new_token
                self.token_issued_at = datetime.now()
                print(f"DEBUG: Token refreshed: {mask_secret(new_token)}")
            else:
                print("ERROR: Token provider returned None")
                
        except Exception as e:
            print(f"ERROR: Token refresh failed: {e}")
            raise TokenExpiredError(f"Failed to refresh token: {e}")
    
    def _validate_messages(self, messages: List[Dict[str, str]]):
        """
        Validate message structure
        
        Reference: "Adding Context" document - system should be first
        
        Args:
            messages: List of message dicts
        
        Raises:
            ValueError if validation fails
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Check if first message is system (recommended)
        if messages[0].get('role') != 'system':
            print("WARNING: First message should be 'system' role for best results")
        
        # Validate all messages have role and content
        for i, msg in enumerate(messages):
            if 'role' not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            if 'content' not in msg:
                raise ValueError(f"Message {i} missing 'content' field")
            if not msg['content']:
                print(f"WARNING: Message {i} has empty content")
            
            # Validate role values
            valid_roles = ['system', 'user', 'assistant']
            if msg['role'] not in valid_roles:
                raise ValueError(f"Message {i} has invalid role '{msg['role']}'. Must be: {valid_roles}")
    
    def _build_payload(self, messages: List[Dict[str, str]], system_message: str = None) -> Dict:
        """
        Build request payload with conversation history management
        
        Reference: "Adding Context" document
        
        Args:
            messages: List of message dicts
            system_message: Optional system message to prepend
        
        Returns:
            Payload dict for Horizon API
        """
        # Extract and organize messages
        sys_msg = system_message
        filtered_messages = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                if not sys_msg:
                    sys_msg = msg['content']
            else:
                filtered_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Default system message
        if not sys_msg:
            sys_msg = "You are a helpful AI assistant for data analysis and Python programming."
        
        # Build messages with system first
        all_messages = [{"role": "system", "content": sys_msg}] + filtered_messages
        
        # Validate schema
        self._validate_messages(all_messages)
        
        # Trim to token limit
        total_tokens = sum(estimate_tokens(msg['content']) for msg in all_messages)
        
        if total_tokens > self.max_context_tokens:
            print(f"WARNING: Total tokens ({total_tokens}) exceeds limit ({self.max_context_tokens})")
            print(f"DEBUG: Trimming conversation history...")
            
            # Use ConversationBuffer to trim
            buffer = ConversationBuffer(self.max_context_tokens)
            for msg in all_messages:
                buffer.add_message(msg['role'], msg['content'])
            
            all_messages = buffer.get_messages()
        
        # Build payload
        payload = {
            "messages": all_messages,
            "stream": False
        }
        
        # Debug output (mask sensitive data)
        print(f"DEBUG: Payload: {len(all_messages)} messages, ~{total_tokens} tokens")
        
        return payload
    
    def _make_request_with_retry(self, payload: Dict) -> requests.Response:
        """
        Make HTTP request with retry logic
        
        Features:
        - Automatic retry on 429 (rate limit) and 5xx (server error)
        - Exponential backoff with jitter
        - Token refresh on 401
        
        Args:
            payload: Request payload
        
        Returns:
            Response object
        
        Raises:
            HorizonAPIError on failure
        """
        url = self.base_url
        
        for attempt in range(MAX_RETRIES):
            # Check if token needs refresh
            if self._should_refresh_token():
                self._refresh_token()
            
            # Build headers
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            try:
                print(f"DEBUG: Request attempt {attempt + 1}/{MAX_RETRIES} to {url}")
                
                start_time = time.time()
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT,
                    verify=False
                )
                elapsed = time.time() - start_time
                
                print(f"DEBUG: Response status {response.status_code} in {elapsed:.2f}s")
                
                # Success
                if response.status_code == 200:
                    return response
                
                # Token expired - try to refresh and retry
                if response.status_code == 401:
                    print("WARNING: 401 Unauthorized - token may be expired")
                    
                    if self.token_provider and attempt < MAX_RETRIES - 1:
                        self._refresh_token()
                        continue
                    else:
                        raise TokenExpiredError("Token expired and no token_provider configured")
                
                # Rate limit - retry with backoff
                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        delay = calculate_retry_delay(attempt)
                        print(f"WARNING: Rate limited (429). Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise RateLimitError(f"Rate limit exceeded after {MAX_RETRIES} retries")
                
                # Server error - retry with backoff
                if 500 <= response.status_code < 600:
                    if attempt < MAX_RETRIES - 1:
                        delay = calculate_retry_delay(attempt)
                        print(f"WARNING: Server error ({response.status_code}). Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise HorizonServerError(f"Server error {response.status_code} after {MAX_RETRIES} retries: {response.text}")
                
                # Other error - don't retry
                raise HorizonAPIError(f"API error {response.status_code}: {response.text}")
                
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES - 1:
                    delay = calculate_retry_delay(attempt)
                    print(f"WARNING: Request timeout. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise HorizonAPIError(f"Request timeout after {MAX_RETRIES} retries")
            
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    delay = calculate_retry_delay(attempt)
                    print(f"WARNING: Request failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise HorizonAPIError(f"Request failed after {MAX_RETRIES} retries: {e}")
        
        raise HorizonAPIError("Maximum retries exceeded")
    
    class ChatCompletion:

        """Chat completion interface for Horizon API"""

        def __init__(self, client):

            self.client = client

            self.completions = self

        def create(self, model: str, messages: List[Dict[str, str]], 

                   stream: bool = False, **kwargs):

            """

            Create chat completion

            Reference: "Text Collection - Chats" document

            Args:

                model: Model name (may not be used by Horizon)

                messages: List of messages with role and content

                stream: Whether to stream response

                **kwargs: Additional parameters

            Returns:

                CompletionResponse object OR Generator (if stream=True)

            """

            # Build payload

            payload = self.client._build_payload(messages, None)

            payload['stream'] = False  # Horizon API doesn't support server-side streaming

            # Make request with retry logic

            response = self.client._make_request_with_retry(payload)

            # Parse response

            try:

                data = response.json()

                if 'message' in data:

                    message = data['message']

                    content = message.get('content', '')

                    print(f"DEBUG: Response received ({len(content)} chars)")

                    # Estimate usage (Horizon may not provide this)

                    usage = UsageStats(

                        prompt_tokens=sum(estimate_tokens(msg['content']) for msg in payload['messages']),

                        completion_tokens=estimate_tokens(content),

                        total_tokens=0  # Will be calculated

                    )

                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

                    # If streaming requested, simulate it

                    if stream:

                        return self._simulate_streaming(content)

                    else:

                        return CompletionResponse(content, usage)

                else:

                    print(f"WARNING: Unexpected response format: {data}")

                    if stream:

                        return self._simulate_streaming(str(data))

                    else:

                        return CompletionResponse(str(data), UsageStats())

            except json.JSONDecodeError as e:

                print(f"ERROR: JSON decode failed: {e}")

                if stream:

                    return self._simulate_streaming(response.text)

                else:

                    return CompletionResponse(response.text, UsageStats())

        def _simulate_streaming(self, content: str):
        
            """
    
            Simulate streaming by yielding chunks of the response
    
            Args:
    
                content: Full response content to stream
    
            Yields:
    
                StreamingChunk objects
    
            """
    
            # Split content into words for natural streaming
    
            words = content.split(' ')
    
            for i, word in enumerate(words):
            
                # Add space before word except for first word
    
                chunk_text = word if i == 0 else ' ' + word
    
                yield StreamingChunk(chunk_text)
 

def create_horizon_client(token: str, 
                          base_url: str = None,
                          token_provider: Optional[Callable[[], str]] = None,
                          max_context_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Factory function to create Horizon client
    
    Args:
        token: Bearer token for authentication
        base_url: Base URL for Horizon API
        token_provider: Optional callback to refresh token
        max_context_tokens: Max tokens for conversation history
    
    Returns:
        HorizonClient instance
    """
    return HorizonClient(token, base_url, token_provider, max_context_tokens)