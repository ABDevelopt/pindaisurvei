with open('run_grid_cropper.py', encoding='utf-8') as f:
    code = f.read()

idx = code.find('text_field')
print('Found text_field in run_grid_cropper.py:', idx)
if idx != -1:
    print(code[idx-100:idx+300])
