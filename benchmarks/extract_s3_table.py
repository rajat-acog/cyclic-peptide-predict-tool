"""Extract Table S3 (63 cyclic monomers with PDB IDs, sequences and HighFold3 RMSD) from the paper's
supplementary .docx into JSON. The paper (Cao et al., Brief. Bioinform. 2025, bbaf488) is CC BY-NC 4.0,
so its data is NOT stored in this repository: download the supplementary file yourself
(Europe PMC PMC12450345 -> supplementary files) and run:

    python -I extract_s3_table.py supplementary_files-250829_bbaf488.docx s3.json
"""
import json, re, sys, zipfile

docx, out = sys.argv[1:3]
xml = zipfile.ZipFile(docx).read('word/document.xml').decode('utf8', 'replace')
rows = []
for tr in re.findall(r'<w:tr[ >].*?</w:tr>', xml, flags=re.S):
    rows.append([''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', tc)).strip()
                 for tc in re.findall(r'<w:tc>.*?</w:tc>', tr, flags=re.S)])
start = [i for i, r in enumerate(rows) if r[:3] == ['PDB ID', 'Sequence', 'Method']][0]
table = []
for r in rows[start + 1:]:
    if r and r[0] == 'PDB ID':
        break
    if len(r) >= 5 and re.match(r'^[1-9][A-Za-z0-9]{3}$', r[0]):
        table.append(dict(pdb=r[0], seq=r[1], method=r[2], hf3_rmsd=float(r[3]), hf3_plddt=float(r[4])))
json.dump(table, open(out, 'w'))
print(len(table), 'peptides')
