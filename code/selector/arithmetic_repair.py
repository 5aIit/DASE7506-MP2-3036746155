"""Arithmetic repair for the frozen selector."""

import ast, math, re

from fractions import Fraction

import numpy as np

from common import parse_answer

from selector.features import _normalize_math, _literal_value, answer_groups

from selector.reasoning_features import LEFT, RIGHT, key, quantities

def signature(node):return ast.dump(node,include_attributes=False)

def repaired_answer(question,text):
    answer=parse_answer(text)
    if answer is None or re.search(r'\b(?:round\w*|approximately|approximate|nearest)\b',text,re.I):return None
    try:target=key(float(Fraction(answer)))
    except (ValueError,ZeroDivisionError,OverflowError):return None
    given=quantities(question);corrections={};final=None;explicit_error=False
    def evaluate(node):
        if signature(node) in corrections:return corrections[signature(node)]
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return float(node.value)
        if isinstance(node,ast.UnaryOp):
            value=evaluate(node.operand)
            if isinstance(node.op,ast.USub):return -value
            if isinstance(node.op,ast.UAdd):return value
        if isinstance(node,ast.BinOp):
            a,b=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
        raise ValueError('Not bounded literal arithmetic')
    for line in _normalize_math(text).splitlines():
        pieces=line.split('=')
        for left,right in zip(pieces,pieces[1:]):
            lm,rm=LEFT.search(left),RIGHT.match(right)
            if not lm or not rm:continue
            if lm.start() and (left[lm.start()-1].isalnum() or left[lm.start()-1] in '_{}^'):continue
            tail=right[rm.end():]
            if tail.startswith(('_','^','%')) or re.match(r'\s*[xyz]\b',tail):continue
            if tail and tail[0].isalnum() and not right[rm.end()-1].isspace():continue
            if re.search(r'%\s*$',left[:lm.start()]):continue
            le,rex=lm.group(1).strip(),rm.group(1).strip();lv,rv=_literal_value(le),_literal_value(rex)
            if lv is None or rv is None:continue
            try:
                lt=ast.parse(le.strip().rstrip('.').strip(),mode='eval').body
                rt=ast.parse(rex.strip().rstrip('.').strip(),mode='eval').body
                correct=evaluate(lt)
            except (ArithmeticError,SyntaxError,ValueError,OverflowError,RecursionError):continue
            if not math.isfinite(correct) or abs(correct)>1e12:continue
            raw_error=abs(lv-rv)/max(1.,abs(lv),abs(rv))
            if raw_error>.01:explicit_error=True
            if not explicit_error:continue
            changed=not math.isclose(correct,rv,rel_tol=1e-8,abs_tol=1e-8)
            if changed and key(rv) not in given:
                corrections[signature(rt)]=correct
            elif not changed:corrections.pop(signature(rt),None)
            if key(rv)==target:final=correct if changed else None
    if final is None or not explicit_error:return None
    candidate=Fraction(str(round(final,10))).limit_denominator(1000000)
    if not math.isclose(float(candidate),final,rel_tol=1e-8,abs_tol=1e-8):return None
    result=str(candidate)
    return result if result!=answer else None

def routes(bank):
    result=[]
    for row,aa in zip(bank,answer_groups(bank)):
        row_routes={}
        for source in set(v for v in aa if v is not None):
            destinations={repaired_answer(row['question'],c['text']) for j,c in enumerate(row['candidates']) if aa[j]==source}
            destinations.discard(None);destinations={v for v in destinations if v in aa and v!=source}
            if len(destinations)==1:row_routes[source]=next(iter(destinations))
        result.append(row_routes)
    return result

def transfer(bank,scores,strength,route=None):
    result=np.asarray(scores,dtype=float).copy()
    if not strength:return result
    if route is None:route=routes(bank)
    for i,(aa,rr) in enumerate(zip(answer_groups(bank),route)):
        keys=list(dict.fromkeys(v for v in aa if v is not None))
        if len(keys)<2:continue
        ids=[[j for j,v in enumerate(aa) if v==a] for a in keys]
        values=np.array([result[i,jj].mean() for jj in ids]);top=np.flatnonzero(np.abs(values-values.max())<=1e-10)
        if len(top)!=1:continue
        source=int(top[0]);destination=rr.get(keys[source])
        if destination is None:continue
        target=keys.index(destination);p=np.exp(values-values.max());p/=p.sum()
        mass=strength*p[source];p[source]-=mass;p[target]+=mass
        for jj,v in zip(ids,np.log(np.maximum(p,1e-12))):result[i,jj]=v
    return result
