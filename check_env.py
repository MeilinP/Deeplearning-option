import sys
import os
import importlib.util

def check_lib(name):
    spec = importlib.util.find_spec(name)
    if spec:
        print(f"✅ {name:<15}: Found at {spec.origin}")
    else:
        print(f"❌ {name:<15}: NOT FOUND")

print("-" * 50)
print(f"Python Executable: {sys.executable}")
print(f"Python Version:    {sys.version.split()[0]}")
print("-" * 50)
print("Checking libraries:")
libs = ["polygon", "matplotlib", "pandas", "numpy", "torch", "dotenv", "tqdm"]
for lib in libs:
    check_lib(lib)
print("-" * 50)
print("System Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("-" * 50)
