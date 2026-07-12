#!/usr/bin/env python3
"""Reproducible empirical-Bayes corner-count prediction using stdlib only."""
import argparse, datetime as dt, json, math, random, statistics, sys
from collections import Counter
from pathlib import Path

class InputError(ValueError): pass

def num(v, name, lo=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v): raise InputError(f"{name} must be numeric")
    v=float(v)
    if lo is not None and v < lo: raise InputError(f"{name} must be >= {lo}")
    return v

def time(v, name):
    try: x=dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception as e: raise InputError(f"{name} must be ISO 8601 with timezone") from e
    if x.tzinfo is None: raise InputError(f"{name} must include timezone")
    return x

def pct(xs, q):
    s=sorted(xs); p=(len(s)-1)*q; a=int(p); b=min(a+1,len(s)-1); return s[a]+(s[b]-s[a])*(p-a)

def summary(xs): return {"mean":round(statistics.fmean(xs),6),"p05":round(pct(xs,.05),6),"p50":round(pct(xs,.5),6),"p95":round(pct(xs,.95),6)}

def poisson(lam, rng):
    if lam < 30:
        limit=math.exp(-lam); k=0; p=1.0
        while p > limit: k+=1; p*=rng.random()
        return k-1
    return max(0, round(rng.gauss(lam, math.sqrt(lam))))

def posterior(base, weight, observations, half, name):
    a=base*weight; b=weight; eff=0.0
    if not isinstance(observations,list) or not observations: raise InputError(f"series.{name} must be non-empty")
    for i,o in enumerate(observations):
        v=num(o.get("value"),f"series.{name}[{i}].value",0)
        if not v.is_integer(): raise InputError(f"series.{name}[{i}].value must be an integer corner count")
        d=num(o.get("days_ago"),f"series.{name}[{i}].days_ago",0); r=num(o.get("reliability",1),f"series.{name}[{i}].reliability",0)
        if r>1: raise InputError("reliability must be <= 1")
        w=r*(.5**(d/half)); a+=w*v; b+=w; eff+=w
    return {"alpha":a,"beta":b,"mean":a/b,"effective_exposure":eff,"raw_count":len(observations)}

def analyze(x):
    m=x.get("match");
    if not isinstance(m,dict): raise InputError("match is required")
    for k in ("id","home_team","away_team","kickoff_utc","as_of"):
        if not isinstance(m.get(k),str) or not m[k]: raise InputError(f"match.{k} is required")
    if time(m["as_of"],"match.as_of") >= time(m["kickoff_utc"],"match.kickoff_utc"): raise InputError("as_of must be before kickoff")
    draws=x.get("draws",20000); seed=x.get("seed",0)
    if isinstance(draws,bool) or not isinstance(draws,int) or draws<1000: raise InputError("draws must be integer >= 1000")
    if isinstance(seed,bool) or not isinstance(seed,int): raise InputError("seed must be integer")
    half=num(x.get("half_life_days"),"half_life_days",0.000001); base=x.get("corner_baseline")
    if not isinstance(base,dict): raise InputError("corner_baseline is required")
    hb=num(base.get("home_rate"),"corner_baseline.home_rate",0.000001); ab=num(base.get("away_rate"),"corner_baseline.away_rate",0.000001); pw=num(base.get("prior_weight"),"corner_baseline.prior_weight",0.000001)
    series=x.get("series",{}); post={
      "home_for":posterior(hb,pw,series.get("home_for"),half,"home_for"),
      "home_against":posterior(ab,pw,series.get("home_against"),half,"home_against"),
      "away_for":posterior(ab,pw,series.get("away_for"),half,"away_for"),
      "away_against":posterior(hb,pw,series.get("away_against"),half,"away_against")}
    lines=x.get("total_lines",[8.5,9.5,10.5,11.5]); lines=[num(v,"total_lines[]",0) for v in lines]
    if any((v*2)%2!=1 for v in lines): raise InputError("total_lines must use half-corner lines")
    rng=random.Random(seed); hs=[]; aws=[]; totals=[]; rates_h=[]; rates_a=[]
    for _ in range(draws):
        z={k:rng.gammavariate(v["alpha"],1/v["beta"]) for k,v in post.items()}
        rh=z["home_for"]*z["away_against"]/hb; ra=z["away_for"]*z["home_against"]/ab
        h=poisson(rh,rng); a=poisson(ra,rng); rates_h.append(rh); rates_a.append(ra); hs.append(h); aws.append(a); totals.append(h+a)
    n=float(draws); c=Counter(totals); top=[{"total":k,"probability":round(v/n,6)} for k,v in c.most_common(5)]
    return {"match":m,"model":{"mode":"empirical_bayes_gamma_poisson","draws":draws,"seed":seed,"half_life_days":half,"baseline":{"home_rate":hb,"away_rate":ab,"prior_weight":pw},"role_posteriors":{k:{q:round(v,6) if isinstance(v,float) else v for q,v in z.items()} for k,z in post.items()},"latent_rates":{"home":summary(rates_h),"away":summary(rates_a)},"predictive_corners":{"home":summary(hs),"away":summary(aws),"total":summary(totals)},"total_lines":{str(v):{"over":round(sum(t>v for t in totals)/n,6),"under":round(sum(t<v for t in totals)/n,6)} for v in lines},"corner_leader":{"home_more":round(sum(h>a for h,a in zip(hs,aws))/n,6),"equal":round(sum(h==a for h,a in zip(hs,aws))/n,6),"away_more":round(sum(h<a for h,a in zip(hs,aws))/n,6)},"top_totals":top},"warnings":["Predictive intervals include parameter and match-count uncertainty but not red cards or unmodelled game-state shocks."]}

def main():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("--output",type=Path); a=p.parse_args()
    try:
        result=analyze(json.loads(a.input.read_text(encoding="utf-8"))); out=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
        a.output.write_text(out,encoding="utf-8") if a.output else print(out,end=""); return 0
    except (OSError,json.JSONDecodeError,InputError) as e: print(f"error: {e}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
