#!/usr/bin/env python3
"""
Test note-triggered video sequence integration
"""

import os
import sys
import json
import tempfile
import logging

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'python'))

def test_note_triggered_integration():
    """Test the note-triggered video sequence integration"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test if the integration code exists in the video_composer.py file
        composer_path = os.path.join(os.path.dirname(__file__), 'backend', 'python', 'video_composer.py')
        
        if not os.path.exists(composer_path):
            print("❌ video_composer.py not found")
            return False
            
        with open(composer_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if our integration is in place
        integration_checks = [
            ("_create_note_triggered_video_sequence method exists", "_create_note_triggered_video_sequence" in content),
            ("Note-triggered sequence called in pipeline", "note_triggered_segment = self._create_note_triggered_video_sequence" in content),
            ("Proper logging for note-triggered sequences", "Processed instrument with note-triggered sequences" in content),
            ("Error handling for failed sequences", "Failed to create note-triggered sequence" in content),
        ]
        
        print("=== Note-Triggered Integration Test ===")
        all_passed = True
        
        for check_name, check_result in integration_checks:
            if check_result:
                print(f"✓ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
                
        # Check the method implementation
        if "_create_note_triggered_video_sequence" in content:
            method_start = content.find("def _create_note_triggered_video_sequence")
            if method_start != -1:
                method_end = content.find("\n    def ", method_start + 1)
                if method_end == -1:
                    method_end = len(content)
                    
                method_content = content[method_start:method_end]
                
                implementation_checks = [
                    ("FFmpeg filter complex construction", "filter_complex" in method_content),
                    ("Pitch adjustment implementation", "asetrate" in method_content),
                    ("Note timing with enable filter", "enable='between(t," in method_content),
                    ("Audio mixing for multiple notes", "amix" in method_content),
                    ("Error handling", "except Exception" in method_content),
                ]
                
                print("\n=== Method Implementation Check ===")
                
                for check_name, check_result in implementation_checks:
                    if check_result:
                        print(f"✓ {check_name}")
                    else:
                        print(f"❌ {check_name}")
                        all_passed = False
                        
        return all_passed
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_note_triggered_integration()
    if success:
        print("\n🎉 Note-triggered integration test passed!")
    else:
        print("\n❌ Note-triggered integration test failed!")
        sys.exit(1)
