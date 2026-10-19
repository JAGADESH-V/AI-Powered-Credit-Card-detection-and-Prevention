#!/usr/bin/env python3
"""Generate the secrets needed for a deployment.

    python scripts/generate_secrets.py >> .env

In production these belong in a secrets manager (AWS Secrets Manager, Vault,
GCP Secret Manager) and should be rotated on a schedule — not in a file.
"""

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptography.fernet import Fernet

print(f"FG_FIELD_ENCRYPTION_KEY={Fernet.generate_key().decode()}")
print(f"FG_JWT_SECRET={secrets.token_urlsafe(48)}")
print(f"FG_HMAC_SIGNING_SECRET={secrets.token_urlsafe(48)}")
