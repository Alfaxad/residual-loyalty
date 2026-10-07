"""Prepare page pairs and geometric diagnostics for human-visible final review."""
from pathlib import Path
import hashlib,json
import fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]

def main():
    rendered=ROOT/'logs/final_render'
    pdf=rendered/'Residual_Loyalty_Comparative_Research_Report.pdf'
    doc=fitz.open(pdf);dest=ROOT/'logs/document_review';dest.mkdir(exist_ok=True)
    for old in rendered.glob('page-*.png'):
        if int(old.stem.split('-')[1])>len(doc):old.unlink()
    for old in dest.glob('pages-*.png'):old.unlink()
    checks=[]
    for i,page in enumerate(doc):
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        outside=[s['text'] for s in spans if s['bbox'][0]<0 or s['bbox'][1]<0 or s['bbox'][2]>page.rect.width+.5 or s['bbox'][3]>page.rect.height+.5]
        checks.append({'page':i+1,'words':len(page.get_text().split()),'outside_page_text':outside})
    for first in range(1,len(doc)+1,2):
        canvas=Image.new('RGB',(1830,1200),'#e1e5e8');draw=ImageDraw.Draw(canvas)
        for col,number in enumerate(range(first,min(first+2,len(doc)+1))):
            page=Image.open(rendered/f'page-{number}.png').convert('RGB')
            page.thumbnail((900,1165),Image.Resampling.LANCZOS)
            canvas.paste(page,(10+col*915,25));draw.text((15+col*915,7),f'Page {number}',fill='black')
        canvas.save(dest/f'pages-{first:02d}-{min(first+1,len(doc)):02d}.png')
    result={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(doc),'page_checks':checks,
            'review_status':'Prepared for visual review; geometric checks do not replace page inspection.'}
    (dest/'geometry.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'pages':len(doc),'outside_text':[c for c in checks if c['outside_page_text']],
                      'sparse_pages':[c for c in checks if c['words']<35]},indent=2))

if __name__=='__main__':main()
