"""Arithmetic rules for the frozen selector."""

import ast, re

from common import parse_answer

from selector.features import _normalize_math, _literal_value, answer_groups

from selector.reasoning_features import LEFT, RIGHT

from selector.arithmetic_repair import repaired_answer

def arithmetic_text(text):
    answer=parse_answer(text)
    if answer is None:return ''
    lines=[]
    for line in _normalize_math(text).replace('TEXT',' ').splitlines():
        pieces=line.split('=')
        for left,right in zip(pieces,pieces[1:]):
            lm,rm=LEFT.search(left),RIGHT.match(right)
            if not lm or not rm:continue
            tail=right[rm.end():]
            if tail.startswith(('_','^','%')) or re.match(r'\s*[xyz]\b',tail):continue
            if tail and tail[0].isalnum() and not right[rm.end()-1].isspace():continue
            if re.search(r'%\s*$',left[:lm.start()]):continue
            le,rex=lm[1].strip(),rm[1].strip()
            if _literal_value(le) is None or _literal_value(rex) is None:continue
            try:node=ast.parse(le.strip().rstrip('.').strip(),mode='eval').body
            except (ValueError,SyntaxError):continue
            if not isinstance(node,ast.BinOp):continue
            if lm.start() and (left[lm.start()-1].isalnum() or left[lm.start()-1] in '_{}^'):continue
            lines.append(le+' = '+rex)
    # Preserve the original parsed final value and rounding abstention signal.
    if re.search(r'\b(?:round\w*|approximately|approximate|nearest)\b',text,re.I):lines.append('rounded')
    return '\n'.join(lines)+f'\n#### {answer}'

def eligible(text):
    checks=[]
    for line in _normalize_math(text).splitlines():
        pieces=line.split('=')
        for left,right in zip(pieces,pieces[1:]):
            lm,rm=LEFT.search(left),RIGHT.match(right)
            if not lm or not rm:continue
            lv,rv=_literal_value(lm[1]),_literal_value(rm[1])
            if lv is None or rv is None:continue
            try:node=ast.parse(rm[1].strip().rstrip('.').strip(),mode='eval').body
            except (ValueError,SyntaxError):continue
            checks.append((abs(lv-rv)/max(1,abs(lv),abs(rv))>.01,isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div)))
    bad=[i for i,(wrong,_) in enumerate(checks) if wrong]
    return len(bad)==1 and checks[bad[0]][1] and bad[0]<len(checks)-1

def routes(bank):
    out=[]
    for row,aa in zip(bank,answer_groups(bank)):
        rr={}
        for source in set(v for v in aa if v is not None):
            texts=[arithmetic_text(c['text']) for j,c in enumerate(row['candidates']) if aa[j]==source]
            ds={repaired_answer(row['question'],text) for text in texts if eligible(text)}
            ds={v for v in ds if v is not None and v in aa and v!=source}
            if len(ds)==1:rr[source]=next(iter(ds))
        out.append(rr)
    return out
