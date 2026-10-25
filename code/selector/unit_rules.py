"""Unit rules for the frozen selector."""

import re

from fractions import Fraction

import numpy as np

from selector.binding_features import _plain, _CANON

from selector.semantic_features import _target

from common import parse_answer

from selector.features import answer_groups

_NUM=r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?'

_STOP=set('a an the of at by to for from as than and or is are was were be been being has have had can could will would should must does did do need needs needed use used uses make makes made cost costs total answer final result left remaining more less fewer additional extra still now then after before per each every all in on with between across into it its his her their our my your equals equal plus minus times divided multiplied since so because which that this these those if only'.split())

_PHYSICAL={
    'length':set('meter metre centimeter centimetre millimeter millimetre kilometer kilometre mile inch foot feet yard cm km mm'.split()),
    'mass':set('gram kilogram milligram ounce pound lb kg mg ton'.split()),
    'volume':set('liter litre milliliter millilitre gallon quart pint cup ml'.split()),
    'time':set('second minute hour day week month year decade sec min hr'.split()),
    'money':set('dollar cent euro peseta money profit saving gain revenue pay payment tip change'.split())}

_PEOPLE=set('person people student pupil child children adult man men woman women boy girl tourist guest worker employee customer patient friend spectator visitor participant'.split())

def unit(word,power=1):
    word=word.lower();word=_CANON.get(word,word)
    if word.endswith('ies'):word=word[:-3]+'y'
    elif len(word)>3 and word.endswith('s') and not word.endswith(('ss','us')):word=word[:-1]
    if word=='in' and power>1:return ('length',power)
    if word in _STOP or len(word)<2:return None
    for family,words in _PHYSICAL.items():
        if word in words:return (family,power)
    if word in _PEOPLE:return ('count:person',1)
    return ('count:'+word,1)

def target_unit(question):
    target=_plain(_target(question))
    match=re.search(r'\bhow\s+many\s+(?:(?:more|fewer|additional|extra|total)\s+)?(?:(square|cubic)\s+)?([a-z]+)',target)
    if match:return unit(match[2],{'square':2,'cubic':3}.get(match[1],1))
    if re.search(r'\bhow\s+much\s+(?:money|profit|change|pay|revenue)\b',target):return ('money',1)
    return None

def result_units(text,answer):
    if answer is None:return set()
    text=_plain(text)
    text=re.sub(r'\\(?:boxed|fbox)\{([^{}]*)\}',r' \1 ',text)
    text=text.replace('\\ ',' ').replace('\\;',' ').replace('\\!',' ')
    # Unit labels must directly follow an occurrence of the official answer.
    pattern=r'(?<![\w.])('+_NUM+r')\s*[$}]*\s*(?:(square|cubic)\s+)?([a-z]+)(?:\s*\^\s*\{?([23])\}?)?'
    found=[]
    for match in re.finditer(pattern,text):
        try:
            if Fraction(match[1].replace(',',''))!=Fraction(answer):continue
        except (ValueError,ZeroDivisionError):continue
        if match.start()<max(0,len(text)-max(400,int(.4*len(text)))):continue
        power=int(match[4]) if match[4] else {'square':2,'cubic':3}.get(match[2],1)
        u=unit(match[3],power)
        if u is not None:found.append((match.end(),u))
    # Prefer the final explicit statement; earlier equal-valued quantities can
    # have other meanings. A final bare boxed value supplies no new unit.
    return {found[-1][1]} if found else set()

def features(bank):
    out=np.zeros((len(bank),4,2),dtype=np.float64)
    for i,row in enumerate(bank):
        target=target_unit(str(row['question']))
        if target is None:continue
        for j,candidate in enumerate(row['candidates']):
            units=result_units(candidate['text'],parse_answer(candidate['text']))
            if len(units)!=1:continue
            observed=next(iter(units))
            if observed==target:continue
            # Cross-count labels and physical dimensions are separate signals.
            out[i,j,0 if not target[0].startswith('count:') and not observed[0].startswith('count:') else 1]=1.
    # Apply a penalty only when every supporting trace has an explicit conflict.
    for i,aa in enumerate(answer_groups(bank)):
        for a in set(v for v in aa if v is not None):
            ids=[j for j,v in enumerate(aa) if v==a];out[i,ids]=out[i,ids].min(0)
    return out
