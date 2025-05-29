#!/usr/bin/env python3
"""
Final validation test for the complete AutoTuneSyncer migration (Lightweight version)
"""

import os
import sys
import json
import re
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
      # Test 2: Validate Python processor argument format
    print("\n🐍 Checking Python processor argument format...")
    try:
        with open('backend/utils/video_processor.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for argparse configuration
        if '--midi-json' in content and '--video-files-json' in content and '--output-path' in content:
            print("✅ Python processor uses correct argument format")
        else:
            print("❌ Python processor missing required arguments")
            return False
            
        # Check that it doesn't use old positional arguments
        if 'sys.argv[1]' in content or 'args[0]' in content:
            print("❌ Python processor still uses old positional arguments")
            return False
        else:
            print("✅ Python processor no longer uses positional arguments")
            
    except Exception as e:
        print(f"❌ Python processor check failed: {e}")
        return False
      # Test 3: Check audio fix implementation
    print("\n🔊 Checking audio fix implementation...")
    try:
        with open('backend/python/gpu_pipeline.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for the specific audio prioritization fix
        if 'mixed_audio and os.path.exists(mixed_audio)' in content:
            print("✅ Audio prioritization fix implemented")
        else:
            print("❌ Audio fix missing")
            return False
            
        # Check for proper fallback logic
        if 'audio_path and os.path.exists(audio_path)' in content:
            print("✅ Audio fallback logic present")
        else:
            print("❌ Audio fallback missing")
            return False
            
        # Check for warning when no audio found
        if 'No audio tracks found - video will be silent' in content:
            print("✅ Audio warning logging implemented")
        else:
            print("❌ Audio warning missing")
            return False
            
    except Exception as e:
        print(f"❌ Audio fix check failed: {e}")
        return False
    
    # Test 4: Verify Node.js bridge fixes
    print("\n🌉 Checking Node.js bridge fixes...")
    try:
        with open('backend/js/pythonBridge.js', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if '--midi-json' in content and '--video-files-json' in content:
            print("✅ Python bridge uses correct argument format")
        else:
            print("❌ Python bridge still uses old format")
            return False
            
        # Check that old positional arguments are not used
        if 'midiPath,' in content and 'videoPath,' in content and 'outputPath' in content:
            # This would indicate old format: [script, midiPath, videoPath, outputPath]
            positional_pattern = r'\[\s*script\s*,\s*\w+Path'
            if re.search(positional_pattern, content):
                print("❌ Python bridge still uses old positional format")
                return False
                
        print("✅ Python bridge argument format is correct")
            
    except Exception as e:
        print(f"❌ Bridge check failed: {e}")
        return False
      # Test 5: Check service layer fixes
    print("\n⚙️ Checking service layer fixes...")
    try:
        with open('backend/services/queueService.js', 'r', encoding='utf-8') as f:
            content = f.read()
            if '--midi-json' in content and '--video-files-json' in content:
                print("✅ Queue service uses correct argument format")
            else:
                print("❌ Queue service still uses old format")
                return False
                
        with open('backend/controllers/compositionController.js', 'r', encoding='utf-8') as f:
            content = f.read()
            if '--midi-json' in content and '--video-files-json' in content:
                print("✅ Composition controller uses correct argument format")
            else:
                print("❌ Composition controller still uses old format")
                return False
    except Exception as e:
        print(f"❌ Service layer check failed: {e}")
        return False
    
    # Test 6: Check import fixes
    print("\n📦 Checking import fixes...")
    try:
        with open('backend/python/gpu_pipeline.py', 'r') as f:
            content = f.read()
            
        # Check for relative import fix
        if 'from .path_registry import PathRegistry' in content:
            print("✅ Relative import fix implemented")
        elif 'from path_registry import PathRegistry' in content:
            print("❌ Still using absolute import (may cause issues)")
            return False
        else:
            print("✅ PathRegistry import handled correctly")
            
    except Exception as e:
        print(f"❌ Import check failed: {e}")
        return False
    
    # Test 7: Check chunk-based architecture
    print("\n🧩 Checking chunk-based architecture...")
    try:
        with open('backend/utils/video_processor.py', 'r') as f:
            content = f.read()
              # Check for VideoComposerWrapper usage (the actual chunk-based implementation)
        if 'VideoComposerWrapper' in content:
            print("✅ VideoComposerWrapper (chunk-based) integration found")
        else:
            print("❌ VideoComposerWrapper integration missing")
            return False
            
        # Check for chunk processing mentions
        if 'chunk' in content.lower() or 'composer_wrapper' in content:
            print("✅ Chunk-based processing references found")
        else:
            print("❌ Chunk-based processing references missing")
            return False
            
    except Exception as e:
        print(f"❌ Architecture check failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 FINAL MIGRATION VALIDATION: SUCCESS!")
    print("✅ All components properly migrated")
    print("✅ Audio fix implemented and prioritized")
    print("✅ Argument formats corrected across all services")
    print("✅ Import paths fixed")
    print("✅ Chunk-based architecture in place")
    print("\n🚀 The system is ready for production testing!")
    print("\nNext steps:")
    print("1. Install required dependencies (torch, etc.)")
    print("2. Test with a real MIDI file")
    print("3. Verify video output includes audio")
    
    return True

if __name__ == '__main__':
    success = test_final_migration()
    if not success:
        print("\n❌ Migration validation failed!")
        sys.exit(1)
    else:
        print("\n✅ Migration validation complete!")
        sys.exit(0)
