"""Structural report checks; run with bundled document Python after final render."""
from pathlib import Path
import datetime, hashlib, json, zipfile
from lxml import etree
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    doc=R/'reports/Residual_Loyalty_Evidence_and_Consequence_Report.docx'
    pdf=doc.with_suffix('.pdf'); errors=[]
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
    with zipfile.ZipFile(doc) as z:
        root=etree.fromstring(z.read('word/document.xml')); styles=etree.fromstring(z.read('word/styles.xml'))
        text=' '.join(root.xpath('//w:t/text()',namespaces=ns))
        for token in ['DEVELOPMENT DRAFT','TODO','TBD','turn0search','turn1search','###','```','<!--']:
            if token in text:errors.append('Unresolved report token: '+token)
        if root.xpath('//w:pBdr',namespaces=ns) or styles.xpath('//w:pBdr',namespaces=ns):errors.append('Paragraph borders remain')
        tables=root.xpath('//w:tbl',namespaces=ns)
        for i,t in enumerate(tables):
            edges=t.xpath('./w:tblPr/w:tblBorders/*',namespaces=ns)
            if len(edges)!=6 or any(e.get('{'+ns['w']+'}color')!='D9D9D9' for e in edges):errors.append('Table border policy failed: '+str(i))
            if not t.xpath('./w:tr[1]/w:trPr/w:tblHeader',namespaces=ns):errors.append('Table header does not repeat: '+str(i))
        images=root.xpath('//wp:docPr',namespaces=ns)
        if any(not im.get('descr') for im in images):errors.append('Image missing alternative text')
    pages=PdfReader(pdf).pages; empty=[]
    for i,page in enumerate(pages,1):
        if len((page.extract_text() or '').strip())<50:empty.append(i)
    if empty:errors.append('Unexpected near-empty PDF pages: '+str(empty))
    result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'document_sha256':sha(doc),'pdf_sha256':sha(pdf),'pages':len(pages),'tables':len(tables),'images_with_alt_text':len(images),'paragraphs':len(root.xpath('//w:p',namespaces=ns)),'scope':'Structure, exact-file identities and extracted text only; separate all-page visual inspection required.','errors':errors,'pass':not errors}
    (R/'logs/document_final_structural_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert not errors
if __name__=='__main__':main()
