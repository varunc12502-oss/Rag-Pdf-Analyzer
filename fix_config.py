import re

with open('app.py', 'r', encoding='utf-8') as f:
    src = f.read()

match = re.search(r'st\.set_page_config\([^)]+\)\n', src, re.DOTALL)
if match:
    config_str = match.group(0)
    src = src.replace(config_str, '')
    
    # Fix the corrupted page_icon if needed
    config_str = re.sub(r'page_icon=.*?,', 'page_icon="📄",', config_str)
    
    insert_point = src.find('CHROMA_PERSIST_DIR')
    src = src[:insert_point] + config_str + '\n' + src[insert_point:]
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(src)
    print('Fixed set_page_config.')
else:
    print('Failed to find set_page_config.')
