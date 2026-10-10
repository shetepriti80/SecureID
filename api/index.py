"""Vercel Serverless Entrypoint for SecureID FastAPI Backend."""

import sys
import os

# Ensure parent directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server import app
