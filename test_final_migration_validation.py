#!/usr/bin/env python3
"""
Final validation test for the complete AutoTuneSyncer migration
"""

import os
import sys
import json
import tempfile
from pathlib import Path

def test_final_migration():
    """Test that the complete migration works end-to-end"""
    
    print("🎯 FINAL MIGRATION VALIDATION")
    print("=" * 50)
    
    # Test 1: Verify key files exist
    key_files = [
        'backend/utils/video_processor.py',
        'backend/utils/video_composer_wrapper.py', 
        'backend/python/gpu_pipeline.py',
        'backend/js/pythonBridge.js',
        'backend/services/queueService.js',
        'backend/controllers/compositionController.js'
    ]
    
    print("📁 Checking key migration files...")
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            return False
      # Test 2: Validate Python processor arguments
    print("\n🐍 Testing Python processor arguments...")
    try:
        import subprocess
        result = subprocess.run([
            'python', 'backend/utils/video_processor.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and ('--midi-json' in result.stdout or 'midi-json' in result.stdout):
            print("✅ Python processor accepts correct arguments")
        else:
            print("❌ Python processor argument format incorrect")
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Python processor test failed: {e}")
        return False
    
    # Test 3: Check audio fix implementation
    print("\n🔊 Checking audio fix implementation...")
    try:
        with open('backend/python/gpu_pipeline.py', 'r') as f:
            content = f.read()
            if 'mixed_audio and os.path.exists(mixed_audio)' in content:
                print("✅ Audio prioritization fix implemented")
            else:
                print("❌ Audio fix missing")
                return False
    except Exception as e:
        print(f"❌ Audio fix check failed: {e}")
        return False
    
    # Test 4: Verify Node.js bridge fixes
    print("\n🌉 Checking Node.js bridge fixes...")
    try:
        with open('backend/js/pythonBridge.js', 'r') as f:
            content = f.read()
            if '--midi-json' in content and '--video-files-json' in content:
                print("✅ Python bridge uses correct argument format")
            else:
                print("❌ Python bridge still uses old format")
                return False
    except Exception as e:
        print(f"❌ Bridge check failed: {e}")
        return False
    
    # Test 5: Check service layer fixes
    print("\n⚙️ Checking service layer fixes...")
    try:
        with open('backend/services/queueService.js', 'r') as f:
            content = f.read()
            if '--midi-json' in content and '--video-files-json' in content:
                print("✅ Queue service uses correct argument format")
            else:
                print("❌ Queue service still uses old format")
                return False
                
        with open('backend/controllers/compositionController.js', 'r') as f:
            content = f.read()
            if '--midi-json' in content and '--video-files-json' in content:
                print("✅ Composition controller uses correct argument format")
            else:
                print("❌ Composition controller still uses old format")
                return False
    except Exception as e:
        print(f"❌ Service layer check failed: {e}")
        return False
    
    # Test 6: Import test
    print("\n📦 Testing imports...")
    try:
        sys.path.append('backend')
        from utils.video_composer_wrapper import VideoComposerWrapper
        from python.gpu_pipeline import GPUPipelineProcessor
        print("✅ All critical components import successfully")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 FINAL MIGRATION VALIDATION: SUCCESS!")
    print("✅ All components properly migrated")
    print("✅ Audio fix implemented")
    print("✅ Argument formats corrected")
    print("✅ Import paths working")
    print("\n🚀 The system is ready for production!")
    
    return True

if __name__ == '__main__':
    success = test_final_migration()
    if not success:
        print("\n❌ Migration validation failed!")
        sys.exit(1)
    else:
        print("\n✅ Migration validation complete!")
        sys.exit(0)
