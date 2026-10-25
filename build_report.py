"""Render REPORT_EN.md as the final English report with embedded fonts.

Requires reportlab; use the bundled document runtime or pip install reportlab.
The report consumes recorded measurements, never scoring references.
"""
import html
import os
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, KeepTogether)
import reportlab

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output/pdf/MP2_Report_Meng_Weitao_3036746155.pdf'
FONT = 'ReportEnglish'
# ReportLab bundles these embeddable fonts, so no system-specific fonts are needed.
font_dir = Path(reportlab.__file__).resolve().parent / 'fonts'
font_path = os.environ.get('MP2_REPORT_FONT', str(font_dir / 'Vera.ttf'))
bold_path = os.environ.get('MP2_REPORT_BOLD_FONT', str(font_dir / 'VeraBd.ttf'))
pdfmetrics.registerFont(TTFont(FONT, font_path))
pdfmetrics.registerFont(TTFont(FONT+'Bold', bold_path))
pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT+'Bold', italic=FONT, boldItalic=FONT+'Bold')
INK = colors.HexColor('#163047')
ACCENT = colors.HexColor('#177e83')
MUTED = colors.HexColor('#5c6975')
LIGHT = colors.HexColor('#e7edef')

STYLES = {
    'body': ParagraphStyle('body', fontName=FONT, fontSize=10, leading=14,
                           textColor=INK, spaceAfter=7,
                           allowWidows=0, allowOrphans=0),
    'title': ParagraphStyle('title', fontName=FONT+'Bold', fontSize=18.2, leading=23,
                            textColor=INK, spaceAfter=13, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName=FONT+'Bold', fontSize=12.5, leading=17,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=8,
                         keepWithNext=True),
    'table': ParagraphStyle('table', fontName=FONT, fontSize=8.6, leading=12,
                            textColor=INK),
    'table_header': ParagraphStyle('table_header', fontName=FONT+'Bold',
                                   fontSize=8.4, leading=11.6, textColor=INK),
    'caption': ParagraphStyle('caption', fontName=FONT, fontSize=9.1, leading=13,
                              textColor=MUTED, spaceAfter=5, keepWithNext=True),
    'code': ParagraphStyle('code', fontName=FONT, fontSize=8.8, leading=12,
                           textColor=MUTED, leftIndent=8),
}


def inline(text):
    # Use predictable ASCII hyphens in prose and arithmetic; leave source intact.
    text = text.translate(str.maketrans({c: '-' for c in '\u2010\u2011\u2012\u2013\u2014\u2212'}))
    text = html.escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<font color="#177e83">\1</font>', text)



def make_table(lines):
    rows=[]
    for line in lines:
        values=[v.strip() for v in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?',v) for v in values):
            continue
        style = STYLES['table_header'] if not rows else STYLES['table']
        rows.append([Paragraph(inline(v), style) for v in values])
    count=len(rows[0])
    widths={4:[236,91,88,92],2:[270,237]}[count]
    if count == 4 and '| Split |' in lines[0]:
        widths = [75,65,85,282]
    table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),LIGHT), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#b1c0c5')),
        ('LINEBELOW',(0,1),(-1,-1),.25,LIGHT),
    ]))
    return table

def page(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.line(44, height - 32, width - 44, height - 32)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7.8)
    canvas.drawString(44, height - 24, 'DASE7506 / MP2 / Meng Weitao / 3036746155')
    canvas.drawString(44, 25, 'Release 1.0 | Hard validation 92.11% | Test 84.58%')
    canvas.drawRightString(width - 44, 25, str(doc.page))
    canvas.restoreState()

class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'h2':
            heading = flowable.getPlainText()
            key = 'section-' + heading.split('.')[0]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(heading, key, level=0)


def main():
    lines = (ROOT / 'REPORT_EN.md').read_text().splitlines()
    assert not any('{{' in line for line in lines), 'Unfilled measured values'
    story = []
    i = 0
    in_front_matter = True
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), STYLES['title']))
            i += 1
            continue
        if line.startswith('## '):
            in_front_matter = False
            story.append(Paragraph(inline(line[3:]), STYLES['h2']))
            i += 1
            continue
        if line.startswith('|'):
            table = []
            while i < len(lines) and lines[i].startswith('|'):
                table.append(lines[i])
                i += 1
            story.append(KeepTogether([make_table(table), Spacer(1, 9)]))
            continue
        if line.startswith('```'):
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                story.append(Paragraph(html.escape(lines[i]), STYLES['code']))
                i += 1
            story.append(Spacer(1, 8))
            i += 1
            continue
        paragraph = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '```')):
            paragraph.append(lines[i])
            i += 1
        style = STYLES['caption'] if in_front_matter else STYLES['body']
        story.append(Paragraph(inline(' '.join(paragraph)), style))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = ReportDocument(str(OUT), pagesize=A4, leftMargin=44, rightMargin=44,
        topMargin=46, bottomMargin=43,
        title='Candidate Comparison and Local Constraints for Mathematical Answer Selection',
        author='Meng Weitao', subject='DASE7506 MP2 - Release 1.0', lang='en')
    document.build(story, onFirstPage=page, onLaterPages=page)
    print(OUT)

if __name__ == "__main__":
    main()
