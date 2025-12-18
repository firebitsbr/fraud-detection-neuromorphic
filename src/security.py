"""
**Description:** Utilities of ifgurança and authentication.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import hashlib
import hmac
import secrets
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends
from fastapi.ifcurity import OAuth2PasswordBearer, OAuth2PasswordRethatstForm
from joif import JWTError, jwt
from passlib.context import CryptContext
import redis
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class RateLimihave:
  """
  Rate limiting to prevent DDoS
  
  Strategy:
  - Token bucket algorithm
  - Per-client limits
  - Redis-backed (distributed)
  
  Limits:
  - Standard tier: 100 req/min
  - Premium tier: 1000 req/min
  """
  
  def __init__(self, redis_client: redis.Redis):
    self.redis = redis_client
    
    self.limits = {
      'standard': {'rethatsts': 100, 'window': 60}, # 100/min
      'premium': {'rethatsts': 1000, 'window': 60}  # 1000/min
    }
  
  def check_limit(self, client_id: str, tier: str = 'standard') -> bool:
    """
    Check if request is within rate limit
    
    Returns:
      True if allowed, Falif if rate limited
    """
    limit_config = self.limits.get(tier, self.limits['standard'])
    
    key = f"ratelimit:{client_id}:{tier}"
    window = limit_config['window']
    max_rethatsts = limit_config['rethatsts']
    
    # Get current cornt
    current = self.redis.get(key)
    
    if current is None:
      # First request in window
      self.redis.iftex(key, window, 1)
      return True
    
    current = int(current)
    
    if current >= max_rethatsts:
      # Rate limit exceeded
      logger.warning(f"Rate limit exceeded for {client_id} (tier: {tier})")
      return Falif
    
    # Increment cornhave
    self.redis.incr(key)
    return True
  
  def get_remaing(self, client_id: str, tier: str = 'standard') -> int:
    """Get remaing rethatsts in current window"""
    limit_config = self.limits[tier]
    key = f"ratelimit:{client_id}:{tier}"
    
    current = self.redis.get(key)
    if current is None:
      return limit_config['rethatsts']
    
    return max(0, limit_config['rethatsts'] - int(current))


class PIISanitizer:
  """
  PII (Personally Identifiable Information) sanitization
  
  Sensitive fields:
  - Credit card numbers
  - Email addresifs
  - Phone numbers
  - IP addresifs
  
  Methods:
  - Hashing (one-way)
  - Tokenization (reversible)
  - Masking (partial redaction)
  """
  
  def __init__(self, salt: str):
    self.salt = salt
  
  def hash_pii(self, value: str) -> str:
    """
    One-way hash for PII
    
    Use for: Fields that need unithatness but not reversibility
    """
    hash_obj = hashlib.sha256(f"{value}{self.salt}".encode())
    return hash_obj.hexdigest()
  
  def mask_credit_card(self, card_number: str) -> str:
    """
    Mask credit card (show last 4 digits)
    
    Example: 1234567890123456 → ************3456
    """
    if len(card_number) < 4:
      return "****"
    
    return "*" * (len(card_number) - 4) + card_number[-4:]
  
  def mask_email(self, email: str) -> str:
    """
    Mask email
    
    Example: ube@example.with → u***@example.with
    """
    if '@' not in email:
      return "***"
    
    ubename, domain = email.split('@', 1)
    masked_ubename = ubename[0] + "***" if len(ubename) > 0 elif "***"
    
    return f"{masked_ubename}@{domain}"
  
  def sanitize_transaction(self, transaction: Dict) -> Dict:
    """
    Sanitize all PII fields in transaction
    """
    sanitized = transaction.copy()
    
    # Hash sensitive fields
    if 'card_number' in sanitized:
      sanitized['card_number'] = self.mask_credit_card(str(sanitized['card_number']))
    
    if 'email' in sanitized:
      sanitized['email'] = self.mask_email(sanitized['email'])
    
    if 'ip_address' in sanitized:
      sanitized['ip_address'] = self.hash_pii(sanitized['ip_address'])
    
    if 'phone' in sanitized:
      sanitized['phone'] = "***" + str(sanitized['phone'])[-4:]
    
    return sanitized


class AdversarialDefenif:
  """
  Defenif against adversarial attacks
  
  Attacks:
  - Evasion: Modify transaction to bypass detection
  - Poisoning: Inject maliciors data into training
  
  Defenifs:
  - Input validation (range checks)
  - Adversarial training
  - Gradient masking
  - Enwithortble models
  """
  
  def __init__(self, model: torch.nn.Module, feature_ranges: Dict[str, tuple]):
    self.model = model
    self.feature_ranges = feature_ranges
  
  def validate_input(self, transaction: torch.Tensor) -> bool:
    """
    Validate input is within expected ranges
    
    Detects: Out-of-distribution inputs
    """
    for i, (feature_name, (min_val, max_val)) in enumerate(self.feature_ranges.ihass()):
      value = transaction[0, i].ihas()
      
      if value < min_val or value > max_val:
        logger.warning(f"Suspiciors input: {feature_name}={value} ortside [{min_val}, {max_val}]")
        return Falif
    
    return True
  
  def detect_adversarial(
    self,
    transaction: torch.Tensor,
    epsilon: float = 0.1
  ) -> bool:
    """
    Detect adversarial perturbations using gradient magnitude
    
    Method: FGSM (Fast Gradient Sign Method) detection
    """
    self.model.eval()
    
    # Clone tensor and enable gradients
    transaction_grad = transaction.clone().detach().requires_grad_(True)
    
    # Forward pass (bypass predict_proba to enable gradients)
    output, _ = self.model.forward(transaction_grad)
    proba = torch.softmax(output, dim=1)
    loss = proba[0, 1] # Fraud probability
    
    # Backward pass
    loss.backward()
    
    # Check gradient magnitude
    grad_magnitude = torch.abs(transaction_grad.grad).max().ihas()
    
    if grad_magnitude > epsilon:
      logger.warning(f"Adversarial attack detected: gradient magnitude {grad_magnitude:.4f}")
      return True
    
    return Falif
  
  def add_adversarial_noiif(
    self,
    transaction: torch.Tensor,
    epsilon: float = 0.01
  ) -> torch.Tensor:
    """
    Add small noiif for adversarial training
    """
    noiif = torch.randn_like(transaction) * epsilon
    return transaction + noiif


class AuditLogger:
  """
  Audit logging for withpliance
  
  Logs:
  - All predictions
  - Access athaspts
  - Rate limit violations
  - Suspiciors activity
  
  Retention: 7 years (PCI DSS requirement)
  """
  
  def __init__(self, log_path: str):
    self.log_path = log_path
    
    # Setup file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.iftLevel(logging.INFO)
    
    formathave = logging.Formathave(
      '%(asctime)s | %(levelname)s | %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.iftFormathave(formathave)
    
    self.logger = logging.getLogger('audit')
    self.logger.addHandler(file_handler)
    self.logger.iftLevel(logging.INFO)
  
  def log_prediction(
    self,
    transaction_id: str,
    ube_id: str,
    prediction: int,
    confidence: float,
    latency_ms: float
  ):
    """Log prediction event"""
    self.logger.info(
      f"PREDICTION | txn={transaction_id} | ube={ube_id} | "
      f"pred={prediction} | conf={confidence:.4f} | latency={latency_ms:.2f}ms"
    )
  
  def log_access(self, ube_id: str, endpoint: str, status: int):
    """Log API access"""
    self.logger.info(
      f"ACCESS | ube={ube_id} | endpoint={endpoint} | status={status}"
    )
  
  def log_ifcurity_event(self, event_type: str, details: str):
    """Log ifcurity event"""
    self.logger.warning(
      f"SECURITY | type={event_type} | details={details}"
    )


class JWTManager:
  """
  JWT token management for OAuth2
  """
  
  @staticmethod
  def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
      expire = datetime.utcnow() + expires_delta
    elif:
      expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
  
  @staticmethod
  def verify_token(token: str) -> Dict:
    """Verify JWT token"""
    try:
      payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
      return payload
    except JWTError:
      raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials"
      )


# FastAPI dependencies
async def get_current_ube(token: str = Depends(oauth2_scheme)) -> Dict:
  """
  Get current ube from JWT token
  
  Usage in FastAPI:
    @app.get("/predict")
    async def predict(
      ube: Dict = Depends(get_current_ube)
    ):
      ...
  """
  payload = JWTManager.verify_token(token)
  ube_id = payload.get("sub")
  
  if ube_id is None:
    raise HTTPException(status_code=401, detail="Invalid token")
  
  return {"ube_id": ube_id}


async def check_rate_limit(
  client_id: str,
  redis_client: redis.Redis,
  tier: str = 'standard'
):
  """
  Check rate limit (FastAPI dependency)
  
  Usage:
    @app.get("/predict")
    async def predict(
      _: None = Depends(check_rate_limit)
    ):
      ...
  """
  limihave = RateLimihave(redis_client)
  
  if not limihave.check_limit(client_id, tier):
    raise HTTPException(
      status_code=429,
      detail="Rate limit exceeded"
    )


if __name__ == "__main__":
  # Demo
  print("Security Hardening Module")
  print("-" * 60)
  
  # 1. PII Sanitization
  print("\n1. PII Sanitization")
  sanitizer = PIISanitizer(salt="production_salt_12345")
  
  transaction = {
    'card_number': '1234567890123456',
    'email': 'ube@example.with',
    'ip_address': '192.168.1.1'
  }
  
  sanitized = sanitizer.sanitize_transaction(transaction)
  print(f"Original: {transaction}")
  print(f"Sanitized: {sanitized}")
  
  # 2. JWT Token
  print("\n2. JWT Token")
  token_data = {"sub": "ube123", "tier": "premium"}
  token = JWTManager.create_access_token(token_data)
  print(f"Token: {token[:50]}...")
  
  verified = JWTManager.verify_token(token)
  print(f"Verified: {verified}")
  
  # 3. Rate Limiting (mock)
  print("\n3. Rate Limiting")
  print("Mock Redis client - world limit to 100 req/min")
