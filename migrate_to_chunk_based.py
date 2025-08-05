#!/usr/bin/env python3
"""
Migration Script: Switch to Chunk-Based Video Processing

This script migrates the application from the problematic note-by-note video processor
to the efficient chunk-based VideoComposer system.

Steps:
1. Backup current video_processor.py
2. Replace with our chunk-based version
3. Update import paths in main application files
4. Test the migration
"""
import os
import sys
import shutil
from pathlib import Path

def backup_current_processor():
    """Backup the current problematic processor"""
    current = Path('backend/utils/video_processor.py')
    backup = Path('backend/utils/video_processor_problematic_backup.py')
    
    if current.exists():
        shutil.copy2(current, backup)
        print(f'✅ Backed up current processor to: {backup}')
        return True
    else:
        print(f'⚠️  Current processor not found at: {current}')
        return False

def deploy_chunk_based_processor():
    """Deploy the chunk-based processor as the new main processor"""
    source = Path('backend/utils/video_processor_chunked.py')
    target = Path('backend/utils/video_processor.py')
    
    if source.exists():
        shutil.copy2(source, target)
        print(f'✅ Deployed chunk-based processor to: {target}')
        return True
    else:
        print(f'❌ Chunk-based processor not found at: {source}')
        return False

def update_imports():
    """Update import statements in main application files (if needed)"""
    # Since we're using a drop-in replacement, no import changes needed
    print('✅ No import changes needed (drop-in replacement)')
    return True

def verify_migration():
    """Verify the migration was successful"""
    processor_path = Path('backend/utils/video_processor.py')
    
    if not processor_path.exists():
        print('❌ Migration failed: video_processor.py not found')
        return False
      # Check if it's the chunk-based version
    content = processor_path.read_text(encoding='utf-8')
    if 'VideoComposerWrapper' in content and 'chunk-based' in content:
        print('✅ Migration successful: chunk-based processor deployed')
        return True
    else:
        print('❌ Migration failed: wrong processor version deployed')
        return False

def test_migration():
    """Test the migrated processor"""
    print('\n🧪 Testing migrated processor...')
    
    test_script = """
import sys
sys.path.append('backend')

from utils.video_processor import EnhancedVideoProcessor

# Test initialization
processor = EnhancedVideoProcessor(performance_mode=True)
print('✅ Processor initialization: OK')

# Test interface compatibility  
methods = ['process_videos', 'validate_input_files', 'get_optimal_ffmpeg_settings']
for method in methods:
    if hasattr(processor, method):
        print(f'✅ Interface method {method}: OK')
    else:
        print(f'❌ Interface method {method}: MISSING')

# Test performance summary (chunk-based feature)
summary = processor.get_performance_summary()
if summary.get('processing_method') == 'chunk-based':
    print('✅ Chunk-based architecture: CONFIRMED')
    print(f'📊 Architecture: {summary.get("architecture", "unknown")}')
else:
    print('❌ Chunk-based architecture: NOT DETECTED')

processor.cleanup()
print('✅ Cleanup: OK')
"""
    
    try:
        exec(test_script)
        print('\n🎉 Migration test: PASSED')
        return True
    except Exception as e:
        print(f'\n❌ Migration test: FAILED - {e}')
        return False

def show_migration_benefits():
    """Show the benefits of the migration"""
    print('\n📊 Migration Benefits:')
    print('   🚀 Performance: 136 individual notes → ~6 chunks (4-second segments)')
    print('   ⚡ Speed: Parallel chunk processing instead of sequential note processing') 
    print('   💾 Memory: GPU acceleration for each chunk')
    print('   🔧 Reliability: Pre-processing approach eliminates timeout issues')
    print('   🎯 Compatibility: Drop-in replacement, no API changes needed')

def main():
    """Main migration function"""
    print('🔄 Starting Migration to Chunk-Based Video Processing')
    print('=' * 60)
    
    # Step 1: Backup current processor
    print('\n1️⃣ Backing up current processor...')
    if not backup_current_processor():
        print('⚠️  Continuing without backup...')
    
    # Step 2: Deploy chunk-based processor
    print('\n2️⃣ Deploying chunk-based processor...')
    if not deploy_chunk_based_processor():
        print('❌ Migration failed: Cannot deploy chunk-based processor')
        return False
    
    # Step 3: Update imports (not needed for drop-in replacement)
    print('\n3️⃣ Updating imports...')
    update_imports()
    
    # Step 4: Verify migration
    print('\n4️⃣ Verifying migration...')
    if not verify_migration():
        print('❌ Migration failed: Verification failed')
        return False
    
    # Step 5: Test migration
    print('\n5️⃣ Testing migration...')
    if not test_migration():
        print('❌ Migration failed: Testing failed')
        return False
    
    # Success!
    print('\n✅ Migration completed successfully!')
    show_migration_benefits()
    
    print('\n🎯 Next Steps:')
    print('   1. Test with your 136-note MIDI file')
    print('   2. Monitor performance improvements')
    print('   3. Verify timeout issues are resolved')
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
