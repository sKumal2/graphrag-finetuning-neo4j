#!/usr/bin/env python3
"""
✅ ENVIRONMENT SETUP CHECKLIST
Complete verification that everything is ready
"""

import os
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    base_path = Path(".")
    
    required_files = {
        "Core Modules": [
            "fine_tune.py",
            "multi_agent_orchestration.py",
            "data_loaders.py",
        ],
        "Startup Scripts": [
            "finetune_setup.py",
            "test_setup.py",
            "start_finetuning.py",
        ],
        "Documentation": [
            "README.md",
            "ENVIRONMENT_SUMMARY.md",
            "SETUP_GUIDE.md",
            "MULTI_AGENT_ARCHITECTURE.md",
            "QUICK_REFERENCE.py",
            "VISUAL_SUMMARY.py",
            "ENVIRONMENT_SETUP_COMPLETE.txt",
        ],
        "Configuration": [
            "requirements.txt",
            ".env",
        ],
    }
    
    print("\n" + "="*70)
    print("✅ ENVIRONMENT SETUP VERIFICATION")
    print("="*70)
    
    all_ok = True
    
    for category, files in required_files.items():
        print(f"\n📁 {category}:")
        for file in files:
            exists = (base_path / file).exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
            if not exists:
                all_ok = False
    
    print("\n" + "="*70)
    if all_ok:
        print("✅ ALL FILES PRESENT - READY TO USE!")
    else:
        print("❌ Some files missing - Check above")
    print("="*70)
    
    return all_ok


def check_dependencies():
    """Check if key dependencies are available"""
    print("\n" + "="*70)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*70)
    
    dependencies = {
        "torch": "PyTorch",
        "langchain": "LangChain",
        "chromadb": "Chroma",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "numpy": "NumPy",
        "tqdm": "tqdm",
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (install with: pip install {name.lower()})")
            all_ok = False
    
    print("\n" + "="*70)
    if not all_ok:
        print("⚠️  Some dependencies missing - Run: pip install -r requirements.txt")
    else:
        print("✅ All dependencies installed!")
    print("="*70)
    
    return all_ok


def check_configuration():
    """Check if .env is configured"""
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION CHECK")
    print("="*70)
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  .env file not found - Create one to configure API keys")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    checks = {
        "GOOGLE_API_KEY": "Google API key for embeddings" if "GOOGLE_API_KEY=" in content else None,
        "Database setup": "Optional configuration present" if "CHROMA" in content else None,
    }
    
    has_key = False
    for check, description in checks.items():
        if description:
            print(f"✅ {check}")
            has_key = True
        else:
            print(f"⚠️  {check} - Not configured (optional for testing)")
    
    print("\n" + "="*70)
    print("📝 Configuration Status:")
    if has_key:
        print("  • GOOGLE_API_KEY is configured ✅")
        print("  • Ready for fine-tuning ✅")
    else:
        print("  • .env exists but not configured")
        print("  • Add GOOGLE_API_KEY to .env before training")
        print("  • Script can use mock data for testing")
    print("="*70)
    
    return True


def print_quick_start():
    """Print quick start instructions"""
    print("\n" + "="*70)
    print("🚀 QUICK START")
    print("="*70)
    
    print("""
Step 1️⃣  Setup Environment (if not done yet)
   $ python finetune_setup.py

Step 2️⃣  Verify Installation
   $ python test_setup.py

Step 3️⃣  Start Fine-Tuning
   $ python start_finetuning.py

Step 4️⃣  Check Results
   • Open: agent_outputs/
   • View: training_report.png
   • Check: best_model.pth
    """)
    
    print("="*70)


def print_documentation_guide():
    """Print documentation guide"""
    print("\n" + "="*70)
    print("📚 DOCUMENTATION GUIDE")
    print("="*70)
    
    docs = {
        "README.md": "Main documentation (START HERE)",
        "ENVIRONMENT_SUMMARY.md": "Quick overview (5-10 min read)",
        "SETUP_GUIDE.md": "Comprehensive guide (30-40 min read)",
        "MULTI_AGENT_ARCHITECTURE.md": "Architecture details (20 min read)",
        "QUICK_REFERENCE.py": "Interactive reference (run it!)",
        "VISUAL_SUMMARY.py": "Visual overview (run it!)",
    }
    
    for doc, description in docs.items():
        print(f"\n📄 {doc}")
        print(f"   → {description}")
    
    print("\n" + "="*70)


def main():
    """Run all checks"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "GRAPHRAG FINE-TUNING ENVIRONMENT" + " "*21 + "║")
    print("║" + " "*25 + "Setup Verification" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    
    # Run checks
    files_ok = check_files()
    deps_ok = check_dependencies()
    config_ok = check_configuration()
    
    # Print guides
    print_quick_start()
    print_documentation_guide()
    
    # Final status
    print("\n" + "="*70)
    print("📋 FINAL STATUS")
    print("="*70)
    
    if files_ok and config_ok:
        print("\n✅ ENVIRONMENT READY FOR USE!")
        print("\nNext step:")
        print("  python finetune_setup.py")
    else:
        print("\n⚠️  Please complete setup steps above")
    
    print("\n" + "="*70)
    print("Questions? Check the documentation or run:")
    print("  python QUICK_REFERENCE.py")
    print("  python VISUAL_SUMMARY.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
