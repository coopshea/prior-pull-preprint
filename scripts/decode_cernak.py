import sys, pandas as pd
from ord_schema.proto import reaction_pb2 as R
d=pd.read_parquet(sys.argv[1])
rows=[]; first=True
def name(c):
    n=None
    for i in c.identifiers:
        if i.type==R.CompoundIdentifier.NAME: n=i.value
    return n
def code(c):
    for i in c.identifiers:
        if i.type==R.CompoundIdentifier.CUSTOM and 'ChemTypeIdx' in i.details: return i.details.split('_')[0],i.value
    return None,None
for b in d['reaction']:
    r=R.Reaction(); r.ParseFromString(b)
    rec={}
    for i in r.identifiers:
        if i.details in('Sample Name_x','coded recipe','experiment'): rec[i.details]=i.value
    for k,inp in r.inputs.items():
        for c in inp.components:
            t,v=code(c)
            if t and t!='Solv': rec[t]=name(c) or v; rec[t+'_code']=v
            elif t=='Solv': rec.setdefault('Solv',set()).add(name(c) or v)
    rec['Solv']='+'.join(sorted(rec.get('Solv',[])))
    rec['temp']=r.conditions.temperature.setpoint.value if r.conditions.HasField('temperature') else None
    vals={}
    for o in r.outcomes:
        for p in o.products:
            for m in p.measurements:
                vals.setdefault(R.ProductMeasurement.ProductMeasurementType.Name(m.type),[]).append(m.percentage.value if m.HasField('percentage') else (m.float_value.value if m.HasField('float_value') else None))
    rec['y']=vals.get('YIELD',[None])[0]; rec['ntypes']=str(sorted(vals))
    if first: print(r.outcomes[0].__str__()[:700]); first=False
    rows.append(rec)
t=pd.DataFrame(rows); t.to_csv(sys.argv[2],index=False)
print(t.shape); print(t.ntypes.value_counts().head().to_dict()); print(t.y.describe().round(2).to_dict())
print({c:t[c].nunique() for c in t.columns if not c.endswith('_code') and c not in('Sample Name_x','coded recipe')})
print(t.groupby(['experiment','temp']).agg(n=('y','size'),L=('Lig','nunique'),C=('Cat','nunique'),B=('Base','nunique'),E=('Elec','nunique'),N=('Nuc','nunique'),S=('Solv','nunique'),ymax=('y','max')).to_string())
