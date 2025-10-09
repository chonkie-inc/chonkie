"""Patch the late.py file to fix numpy bug"""

import re
import os

def fix_late_chunker():
    """Apply the fix to late.py file"""
    try:
        # Use the correct path - late.py is in the SAME directory as this script
        file_path = 'late.py'
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("🔧 Applying fixes to late.py...")
        
        # 1. Add numpy import at top
        if 'import numpy' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'import importlib.util as importutil' in line:
                    lines.insert(i + 1, 'import numpy')
                    print("✅ Added numpy import")
                    break
            content = '\n'.join(lines)
        
        # 2. Remove TYPE_CHECKING numpy stub
        if 'if TYPE_CHECKING:' in content:
            # Simple removal of the TYPE_CHECKING block
            lines = content.split('\n')
            new_lines = []
            skip = False
            
            for line in lines:
                if 'if TYPE_CHECKING:' in line:
                    skip = True
                    continue
                elif skip and line.strip() == '':
                    continue
                elif skip and 'class np:' in line:
                    continue
                elif skip and '"""Stub class for numpy when not available."""' in line:
                    continue
                elif skip and line.strip() == 'pass':
                    skip = False
                    continue
                elif skip:
                    # Skip other lines in the TYPE_CHECKING block
                    continue
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
            print("✅ Removed TYPE_CHECKING numpy stub")
        
        # 3. Replace np. with numpy. in _get_late_embeddings method
        content = content.replace('np.cumsum', 'numpy.cumsum')
        content = content.replace('np.mean', 'numpy.mean')
        content = content.replace('"np.ndarray"', '"numpy.ndarray"')
        
        # Remove type ignore comments
        content = content.replace('  # type: ignore[name-defined]', '')
        
        print("✅ Fixed _get_late_embeddings method")
        
        # 4. Fix _import_dependencies method
        old_lines = [
            'if importutil.find_spec("numpy"):',
            '    global np',
            '    import numpy as np'
        ]
        
        for old_line in old_lines:
            if old_line in content:
                if 'if importutil.find_spec("numpy"):' in old_line:
                    content = content.replace(old_line, 'if not importutil.find_spec("numpy"):')
                else:
                    content = content.replace(old_line, '')
        
        print("✅ Fixed _import_dependencies method")
        
        # Write fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ SUCCESS: late.py has been patched!")
        return True
        
    except Exception as e:
        print(f"❌ Error patching file: {e}")
        return False

def verify_fix():
    """Verify the fix was applied correctly"""
    try:
        with open('late.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('import numpy' in content, 'numpy import present'),
            ('np.cumsum' not in content, 'np.cumsum removed'),
            ('np.mean' not in content, 'np.mean removed'),
            ('numpy.cumsum' in content, 'numpy.cumsum added'),
            ('numpy.mean' in content, 'numpy.mean added'),
        ]
        
        print("\n🔍 Verification Results:")
        all_passed = True
        for check, description in checks:
            if check:
                print(f"  ✅ {description}")
            else:
                print(f"  ❌ {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Patching LateChunker to fix numpy bug...")
    print("=" * 50)
    
    if fix_late_chunker():
        print("\n" + "=" * 50)
        print("🔍 Verifying the fix...")
        if verify_fix():
            print("\n🎉 SUCCESS: Patch applied and verified!")
        else:
            print("\n⚠️  Patch applied but verification failed!")
    else:
        print("\n💥 Patch failed!")