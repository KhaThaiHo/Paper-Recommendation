import os
import json
from pathlib import Path

import config

print('MODEL_NAME:', config.MODEL_NAME)

home = Path.home()
cache_hub = home / '.cache' / 'huggingface' / 'hub'
cache_transformers = home / '.cache' / 'huggingface' / 'transformers'

print('\nChecking cache directories:')
print(' hub:', cache_hub)
print(' transformers:', cache_transformers)

for p in (cache_hub, cache_transformers):
    if p.exists():
        print(f'Contents of {p} (top-level dirs):')
        try:
            for child in sorted(p.iterdir()):
                print(' -', child.name)
        except Exception as e:
            print('  (error listing)', e)
    else:
        print(f'  {p} does not exist')

print('\nSearching for model name fragments in hub cache...')
frags = [part for part in config.MODEL_NAME.replace('/', ' ').split() if part]
found = False
if cache_hub.exists():
    for root, dirs, files in os.walk(cache_hub):
        for d in dirs:
            name = d.lower()
            if any(frag.lower() in name for frag in frags):
                print('Found folder:', os.path.join(root, d))
                found = True

if not found:
    print('No matching folder found in hub cache')

print('\nTrying to load tokenizer/config locally (no internet)...')
try:
    from transformers import AutoConfig, AutoTokenizer
    try:
        AutoConfig.from_pretrained(config.MODEL_NAME, local_files_only=True)
        print('AutoConfig: OK (found locally)')
    except Exception as e:
        print('AutoConfig: FAIL ->', repr(e))
    try:
        AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
        print('AutoTokenizer: OK (found locally)')
    except Exception as e:
        print('AutoTokenizer: FAIL ->', repr(e))
except Exception as e:
    print('Transformers not available or failed to import:', repr(e))

print('\nDone')
