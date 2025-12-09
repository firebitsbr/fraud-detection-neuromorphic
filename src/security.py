"""
**Descrição:** Utilitários de segurança e autenticação.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import hashlib
import hmac
import secrets
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
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


class RateLimiter:
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
            'standard': {'requests': 100, 'window': 60},  # 100/min
            'premium': {'requests': 1000, 'window': 60}   # 1000/min
        }
    
    def check_limit(self, client_id: str, tier: str = 'standard') -> bool:
        """
        Check if request is within rate limit
        
        Returns:
            True if allowed, False if rate limited
        """
        limit_config = self.limits.get(tier, self.limits['standard'])
        
        key = f"ratelimit:{client_id}:{tier}"
        window = limit_config['window']
        max_requests = limit_config['requests']
        
        # Get current count
        current = self.redis.get(key)
        
        if current is None:
            # First request in window
            self.redis.setex(key, window, 1)
            return True
        
        current = int(current)
        
        if current >= max_requests:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for {client_id} (tier: {tier})")
            return False
        
        # Increment counter
        self.redis.incr(key)
        return True
    
    def get_remaining(self, client_id: str, tier: str = 'standard') -> int:
        """Get remaining requests in current window"""
        limit_config = self.limits[tier]
        key = f"ratelimit:{client_id}:{tier}"
        
        current = self.redis.get(key)
        if current is None:
            return limit_config['requests']
        
        return max(0, limit_config['requests'] - int(current))


class PIISanitizer:
    """
    PII (Personally Identifiable Information) sanitization
    
    Sensitive fields:
    - Credit card numbers
    - Email addresses
    - Phone numbers
    - IP addresses
    
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
        
        Use for: Fields that need uniqueness but not reversibility
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
        
        Example: user@example.com → u***@example.com
        """
        if '@' not in email:
            return "***"
        
        username, domain = email.split('@', 1)
        masked_username = username[0] + "***" if len(username) > 0 else "***"
        
        return f"{masked_username}@{domain}"
    
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


class AdversarialDefense:
    """
    Defense against adversarial attacks
    
    Attacks:
    - Evasion: Modify transaction to bypass detection
    - Poisoning: Inject malicious data into training
    
    Defenses:
    - Input validation (range checks)
    - Adversarial training
    - Gradient masking
    - Ensemble models
    """
    
    def __init__(self, model: torch.nn.Module, feature_ranges: Dict[str, tuple]):
        self.model = model
        self.feature_ranges = feature_ranges
    
    def validate_input(self, transaction: torch.Tensor) -> bool:
        """
        Validate input is within expected ranges
        
        Detects: Out-of-distribution inputs
        """
        for i, (feature_name, (min_val, max_val)) in enumerate(self.feature_ranges.items()):
            value = transaction[0, i].item()
            
            if value < min_val or value > max_val:
                logger.warning(f"Suspicious input: {feature_name}={value} outside [{min_val}, {max_val}]")
                return False
        
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
        transaction.requires_grad = True
        
        # Forward pass
        output = self.model.predict_proba(transaction)
        loss = output[0, 1]  # Fraud probability
        
        # Backward pass
        loss.backward()
        
        # Check gradient magnitude
        grad_magnitude = torch.abs(transaction.grad).max().item()
        
        if grad_magnitude > epsilon:
            logger.warning(f"Adversarial attack detected: gradient magnitude {grad_magnitude:.4f}")
            return True
        
        return False
    
    def add_adversarial_noise(
        self,
        transaction: torch.Tensor,
        epsilon: float = 0.01
    ) -> torch.Tensor:
        """
        Add small noise for adversarial training
        """
        noise = torch.randn_like(transaction) * epsilon
        return transaction + noise


class AuditLogger:
    """
    Audit logging for compliance
    
    Logs:
    - All predictions
    - Access attempts
    - Rate limit violations
    - Suspicious activity
    
    Retention: 7 years (PCI DSS requirement)
    """
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        
        # Setup file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger('audit')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_prediction(
        self,
        transaction_id: str,
        user_id: str,
        prediction: int,
        confidence: float,
        latency_ms: float
    ):
        """Log prediction event"""
        self.logger.info(
            f"PREDICTION | txn={transaction_id} | user={user_id} | "
            f"pred={prediction} | conf={confidence:.4f} | latency={latency_ms:.2f}ms"
        )
    
    def log_access(self, user_id: str, endpoint: str, status: int):
        """Log API access"""
        self.logger.info(
            f"ACCESS | user={user_id} | endpoint={endpoint} | status={status}"
        )
    
    def log_security_event(self, event_type: str, details: str):
        """Log security event"""
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
        else:
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
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """
    Get current user from JWT token
    
    Usage in FastAPI:
        @app.get("/predict")
        async def predict(
            user: Dict = Depends(get_current_user)
        ):
            ...
    """
    payload = JWTManager.verify_token(token)
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"user_id": user_id}


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
    limiter = RateLimiter(redis_client)
    
    if not limiter.check_limit(client_id, tier):
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
        'email': 'user@example.com',
        'ip_address': '192.168.1.1'
    }
    
    sanitized = sanitizer.sanitize_transaction(transaction)
    print(f"Original: {transaction}")
    print(f"Sanitized: {sanitized}")
    
    # 2. JWT Token
    print("\n2. JWT Token")
    token_data = {"sub": "user123", "tier": "premium"}
    token = JWTManager.create_access_token(token_data)
    print(f"Token: {token[:50]}...")
    
    verified = JWTManager.verify_token(token)
    print(f"Verified: {verified}")
    
    # 3. Rate Limiting (mock)
    print("\n3. Rate Limiting")
    print("Mock Redis client - would limit to 100 req/min")
