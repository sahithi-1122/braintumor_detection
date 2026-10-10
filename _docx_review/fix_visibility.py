from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

source_path = Path('Brain_Tumor_Classification_Project_Paper.docx')
path = Path('Brain_Tumor_Classification_Project_Paper_Readable.docx')
temp = path.with_suffix('.tmp.docx')
replacements = {b'D9D9D9': b'000000', b'808080': b'000000'}
changed = []

with ZipFile(source_path, 'r') as source, ZipFile(temp, 'w', ZIP_DEFLATED) as target:
    for item in source.infolist():
        data = source.read(item.filename)
        if item.filename.endswith('.xml'):
            updated = data
            for before, after in replacements.items():
                updated = updated.replace(before, after)
            if updated != data:
                changed.append(item.filename)
                data = updated
        target.writestr(item, data)

temp.replace(path)
print('Normalized pale text colors to black in:', ', '.join(changed))

with ZipFile(path) as check:
    assert check.testzip() is None
    for name in changed:
        payload = check.read(name)
        assert b'D9D9D9' not in payload and b'808080' not in payload
print('DOCX archive verified.')
