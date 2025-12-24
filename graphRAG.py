"""
GraphRAG - Fine-tuning with Multi-Agent Orchestration and Neo4j Integration
This is the main entry point for the GraphRAG system.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify required environment variables
required_vars = ["GOOGLE_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
    print(f"   Please add them to your .env file")
    sys.exit(1)

print("✓ GraphRAG system initialized")
print("  - Google API configured")
print("\nTo get started:")
print("  1. Run: python finetune_setup.py")
print("  2. Run: python example_multi_agent_finetune.py")
print("  3. Read: README.md for full documentation")
