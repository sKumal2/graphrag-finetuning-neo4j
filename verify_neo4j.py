#!/usr/bin/env python
"""
Neo4j Connection Verification Script

Run this to verify your Neo4j setup is correct:
    python verify_neo4j.py
"""

import os
from pathlib import Path
import sys

def check_neo4j_setup():
    """Verify Neo4j setup and configuration"""
    
    print("=" * 60)
    print("NEO4J SETUP VERIFICATION")
    print("=" * 60)
    
    # Step 1: Check if neo4j package is installed
    print("\n1. Checking neo4j package...")
    try:
        import neo4j
        print(f"   ✓ neo4j version {neo4j.__version__} installed")
    except ImportError:
        print("   ✗ neo4j not installed")
        print("   Install with: pip install neo4j")
        return False
    
    # Step 2: Check .env file
    print("\n2. Checking .env file...")
    env_path = Path(".env")
    if not env_path.exists():
        print("   ✗ .env file not found")
        print(f"   Create one: touch {env_path}")
        return False
    print("   ✓ .env file found")
    
    # Step 3: Check for Neo4j credentials
    print("\n3. Checking Neo4j credentials in .env...")
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(str(value)) > 10 else value
            print(f"   ✓ {var}: {masked}")
        else:
            print(f"   ✗ {var}: NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n   Missing credentials: {', '.join(missing)}")
        print("   Add them to .env file:")
        print("""
   NEO4J_URI=neo4j+s://your-db.databases.neo4j.io:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-password
   NEO4J_DATABASE=neo4j
        """)
        return False
    
    # Step 4: Test connection
    print("\n4. Testing Neo4j connection...")
    try:
        from data_loaders import setup_neo4j_from_env
        db = setup_neo4j_from_env()
        
        if db is None:
            print("   ✗ Could not initialize Neo4j")
            return False
        
        # Try to get stats
        stats = db.get_graph_stats()
        print(f"   ✓ Connected successfully!")
        print(f"     - Total documents: {stats['total_nodes']}")
        print(f"     - Total relationships: {stats['total_relationships']}")
        
        db.close()
        
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False
    
    # All checks passed
    print("\n" + "=" * 60)
    print("✓ NEO4J SETUP IS VALID")
    print("=" * 60)
    print("""
Next steps:
1. Run: python example_neo4j_integration.py
2. Read: NEO4J_SETUP_GUIDE.md
3. Use in your pipeline with: setup_neo4j_from_env()
    """)
    return True


def main():
    """Run verification"""
    try:
        success = check_neo4j_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
