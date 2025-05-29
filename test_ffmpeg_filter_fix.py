#!/usr/bin/env python3
"""
Test script to verify the FFmpeg filter syntax fix for video combination
"""

import sys
import os

# Add the backend paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'utils'))

def test_filter_generation():
    """Test the fixed _generate_combination_filter method"""
    print("🔧 Testing FFmpeg Filter Generation Fix")
    print("=" * 50)
    
    try:
        from video_processor import EnhancedVideoProcessor
        
        # Create processor instance
        processor = EnhancedVideoProcessor()
        
        # Test scenarios
        test_cases = [
            {
                'name': '2 videos',
                'input_files': ['video1.mp4', 'video2.mp4'],
                'expected_contains': ['hstack=inputs=2', 'amix=inputs=2']
            },
            {
                'name': '4 videos', 
                'input_files': ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4'],
                'expected_contains': ['vstack=inputs=2', 'amix=inputs=4']
            },
            {
                'name': '8 videos (the problematic case)',
                'input_files': [f'video{i}.mp4' for i in range(1, 9)],
                'expected_contains': ['amix=inputs=8', '[0:a][1:a][2:a][3:a][4:a][5:a][6:a][7:a]']
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            print(f"\n📹 Testing {test_case['name']}:")
            
            # Generate filter
            filter_complex = processor._generate_combination_filter(
                test_case['input_files'], 
                {}  # Empty MIDI data for test
            )
            
            print(f"Generated filter: {filter_complex}")
            
            # Check for expected content
            passed = True
            for expected in test_case['expected_contains']:
                if expected in filter_complex:
                    print(f"  ✅ Contains: {expected}")
                else:
                    print(f"  ❌ Missing: {expected}")
                    passed = False
            
            # Check for problematic syntax (the bug we fixed)
            if '+[' in filter_complex and 'amix' in filter_complex:
                print(f"  ❌ STILL HAS BUG: Found '+[' in audio mix syntax")
                passed = False
            else:
                print(f"  ✅ No '+[' syntax bug found")
            
            if passed:
                print(f"  ✅ {test_case['name']}: PASSED")
            else:
                print(f"  ❌ {test_case['name']}: FAILED")
                all_passed = False
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ALL FILTER TESTS PASSED!")
            print("✅ The FFmpeg filter syntax error has been fixed")
        else:
            print("❌ SOME FILTER TESTS FAILED")
            print("🔧 Additional fixes may be needed")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Filter test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_syntax_specifically():
    """Test specifically the audio mixing syntax that was broken"""
    print("\n🎵 Testing Audio Mixing Syntax Fix")
    print("=" * 50)
    
    try:
        from video_processor import EnhancedVideoProcessor
        
        processor = EnhancedVideoProcessor()
        
        # Test the 8-video case that was failing
        input_files = [f'track{i}.mp4' for i in range(8)]
        filter_complex = processor._generate_combination_filter(input_files, {})
        
        print(f"8-video filter: {filter_complex}")
        
        # Extract audio part
        parts = filter_complex.split(';')
        audio_part = None
        for part in parts:
            if 'amix=inputs=8' in part:
                audio_part = part
                break
        
        if audio_part:
            print(f"Audio mixing part: {audio_part}")
            
            # Check the specific error that was happening
            if '[0:a]+[1:a]+[2:a]' in audio_part:
                print("❌ STILL BROKEN: Found '+' separators in audio mix")
                return False
            elif '[0:a][1:a][2:a][3:a][4:a][5:a][6:a][7:a]amix=inputs=8' in audio_part:
                print("✅ FIXED: Correct audio mixing syntax (no '+' separators)")
                return True
            else:
                print(f"⚠️  Unknown audio syntax: {audio_part}")
                return False
        else:
            print("❌ No audio mixing part found")
            return False
            
    except Exception as e:
        print(f"❌ Audio syntax test error: {e}")
        return False

if __name__ == "__main__":
    print("🎬 FFmpeg Filter Syntax Fix Test")
    print("Testing the fix for: [AVFilterGraph] Error parsing filterchain")
    print("=" * 60)
    
    success1 = test_filter_generation()
    success2 = test_audio_syntax_specifically()
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULT")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 SUCCESS! FFmpeg filter syntax has been fixed!")
        print("✅ The '[0:a]+[1:a]+...' error should be resolved")
        print("🎬 Video combination should now work for 8+ videos")
    else:
        print("❌ FAILED! FFmpeg filter syntax still has issues")
        print("🔧 Additional debugging needed")
    
    print("\nNext steps:")
    print("1. Test with actual video composition")
    print("2. Run full integration test")
    print("3. Verify final video output")
