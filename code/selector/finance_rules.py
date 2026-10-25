"""Finance rules for the frozen selector."""

import re

from fractions import Fraction

import numpy as np

from selector.binding_features import _plain, _unit

from selector.semantic_features import _target

from selector.features import answer_groups

NUM=r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?'

NOUN=r'[a-z]+(?:\s+of\s+[a-z]+)?'

def value(v):return Fraction(v.replace(',',''))

def covered(text,spans):
    return all(any(a<=m.start() and m.end()<=b for a,b in spans)
        for m in re.finditer(r'(?<![\w.])'+NUM,text))

def price_constraint(question):
    text=_plain(question);target=_target(text)
    if not re.search(r'\b(?:paid|pay|pays|cost|costs|price)\b',target):return None
    base=list(re.finditer(r'\b(?:initial|original|regular|full)\s+price\s+(?:was|is|of|=)\s*[$€£]\s*('+NUM+r')',text))
    percent=list(re.finditer(r'\b(?:at|for)\s+('+NUM+r')\s*%\s+(of|off)\s+(?:the\s+)?(?:initial|original|regular|full)\s+price',text))
    if len(base)!=1 or len(percent)!=1:return None
    if not covered(text,[base[0].span(),percent[0].span()]):return None
    rate=value(percent[0][1])/100
    if not 0<=rate<=1:return None
    return str(value(base[0][1])*(rate if percent[0][2]=='of' else 1-rate))

def profit_constraint(question):
    text=_plain(question);target=_target(text)
    if not re.search(r'\b(?:profit|profits|gain|gains)\b',target) or '%' in text:return None
    # Production rates and unit purchase/sale prices are typed by item noun.
    cost_pattern=r'\b(?:each|every|a|an)\s+('+NOUN+r')\s+(?:costs?|is priced at)\s*[$€£]\s*('+NUM+r')'
    sale_pattern=r'\b(?:sell|sells|sold|selling)\s+(?:each|every|a|an)\s+('+NOUN+r')\s+for\s*[$€£]\s*('+NUM+r')'
    rate_pattern=r'\b(?:each|every)\s+('+NOUN+r')\s+(?:takes|uses|requires|needs|contains)\s+('+NUM+r')\s+('+NOUN+r')'
    cost_matches=list(re.finditer(cost_pattern,text));sales=list(re.finditer(sale_pattern,text));rates=list(re.finditer(rate_pattern,text))
    if not cost_matches or len(sales)!=1:return None
    root=_unit(sales[0][1]);selling=value(sales[0][2]);costs={};edges={};spans=[m.span() for m in cost_matches+sales+rates]
    if root is None:return None
    for m in cost_matches:
        k=_unit(m[1]);v=value(m[2])
        if k is None or (k in costs and costs[k]!=v):return None
        costs[k]=v
    for m in rates:
        a,b=_unit(m[1]),_unit(m[3]);n=value(m[2])
        if a is None or b is None or a==b or n<=0:return None
        if b in edges.get(a,{}) and edges[a][b]!=n:return None
        edges.setdefault(a,{})[b]=n
    quantities=[]
    for m in re.finditer(r'(?<![\w.])('+NUM+r')\s+('+NOUN+r')',text):
        if _unit(m[2])==root:
            quantities.append(value(m[1]));spans.append(m.span())
    if len(quantities)!=1 or quantities[0]<=0:return None
    if not covered(text,spans):return None
    # Every explicit monetary amount must belong to a recognized unit price.
    if any(not any(a<=m.start() and m.end()<=b for a,b in spans)
           for m in re.finditer(r'[$€£]\s*'+NUM,text)):return None
    visited=set();active=set()
    def production_cost(item):
        if item in active:raise ValueError('Cyclic production relation')
        active.add(item);visited.add(item)
        if item in edges:
            computed=sum((n*production_cost(resource) for resource,n in edges[item].items()),Fraction(0))
            if item in costs and costs[item]!=computed:raise ValueError('Conflicting unit costs')
        elif item in costs:computed=costs[item]
        else:raise ValueError('Unpriced resource')
        active.remove(item);return computed
    try:unit_cost=production_cost(root)
    except ValueError:return None
    if (set(costs)|set(edges))-visited:return None
    return str(quantities[0]*(selling-unit_cost))

def features(bank):
    out=np.zeros((len(bank),4,2),float)
    for i,(row,aa) in enumerate(zip(bank,answer_groups(bank))):
        for k,solver in enumerate((price_constraint,profit_constraint)):
            try:expected=solver(row['question'])
            except (ValueError,ZeroDivisionError,OverflowError):expected=None
            if expected is None or expected not in aa:continue
            out[i,:,k]=np.array([a==expected for a in aa],float)
    return out
