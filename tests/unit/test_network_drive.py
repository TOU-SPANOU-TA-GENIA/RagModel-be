# test_network_drive.py
"""
Quick test to verify Z: drive is accessible from Python
"""

from pathlib import Path
import sys

print("=" * 60)
print("NETWORK DRIVE ACCESS TEST")
print("=" * 60)

# Test 1: Check if Z: exists
print("\n1. Testing Z: drive access...")
z_drive = Path("Z:")

if z_drive.exists():
    print("   ✅ Z: drive is accessible")
else:
    print("   ❌ Z: drive NOT accessible")
    sys.exit(1)

# Test 2: List contents
print("\n2. Listing Z: drive contents...")
try:
    items = list(z_drive.iterdir())
    print(f"   ✅ Found {len(items)} items:")
    for item in items[:10]:  # Show first 10
        item_type = "📁" if item.is_dir() else "📄"
        print(f"      {item_type} {item.name}")
    if len(items) > 10:
        print(f"      ... and {len(items) - 10} more")
except Exception as e:
    print(f"   ❌ Error listing contents: {e}")
    sys.exit(1)

# Test 3: Count total files
print("\n3. Counting all files recursively...")
try:
    all_files = list(z_drive.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    dirs_only = [f for f in all_files if f.is_dir()]
    
    print(f"   ✅ Total items: {len(all_files)}")
    print(f"      📁 Directories: {len(dirs_only)}")
    print(f"      📄 Files: {len(files_only)}")
except Exception as e:
    print(f"   ❌ Error counting files: {e}")
    sys.exit(1)

# Test 4: Check file types
print("\n4. Analyzing file types...")
try:
    extensions = {}
    for file in files_only:
        ext = file.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print(f"   ✅ File types found:")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]:
        ext_name = ext if ext else "(no extension)"
        print(f"      {ext_name}: {count} files")
except Exception as e:
    print(f"   ❌ Error analyzing files: {e}")

# Test 5: Try reading a file
print("\n5. Testing file read access...")
try:
    # Find any .txt file
    txt_files = [f for f in files_only if f.suffix.lower() == '.txt']
    if txt_files:
        test_file = txt_files[0]
        content = test_file.read_text(encoding='utf-8', errors='ignore')
        print(f"   ✅ Successfully read: {test_file.name}")
        print(f"      Size: {len(content)} characters")
        print(f"      First 100 chars: {content[:100]}...")
    else:
        print("   ⚠️  No .txt files found to test")
except Exception as e:
    print(f"   ❌ Error reading file: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE - Z: drive is ready for RagModel-be!")
print("=" * 60)