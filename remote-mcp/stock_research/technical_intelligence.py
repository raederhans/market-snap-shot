"""Experimental daily technical research: deterministic, causal, no trading.

Use with canonical fetch_bars and require_ohlcv; no new provider or credentials.
All thresholds are our configurable research heuristics, not proprietary ratings.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .core import require_ohlcv, finite_or_none, DataQualityError

VERSION = '0.2.0'
NAMES = {'technical_features', 'pattern_scan', 'stage_analysis', 'trend_template', 'vcp_analysis', 'technical_intelligence'}


def _f(x): return finite_or_none(x)
def _iso(t): return pd.Timestamp(t).isoformat()
def _ratio(a,b): return float(a/b) if np.isfinite(a) and np.isfinite(b) and b != 0 else None

def wilder(s, n=14):
    """Seed from first n consecutive finite observations; restart after a gap."""
    out = np.full(len(s), np.nan); buf=[]; val=None
    for i,x in enumerate(s.to_numpy(dtype=float)):
        if not np.isfinite(x): buf=[]; val=None; continue
        if val is None:
            buf.append(x)
            if len(buf)<n: continue
            val=float(np.mean(buf)); buf=[]
        else: val=(val*(n-1)+x)/n
        out[i]=val
    return pd.Series(out,index=s.index)


def feature_series(d):
    c,h,l,v=d.close,d.high,d.low,d.volume
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=wilder(tr); up=h.diff(); down=-l.diff()
    pos=up.where((up>down)&(up>0),0.).where(up.notna()); neg=down.where((down>up)&(down>0),0.).where(down.notna())
    # DMI seed aligns with the first defined directional move; flat true range is unavailable.
    atr_dm=wilder(tr.where(up.notna())).replace(0,np.nan)
    pdi=100*wilder(pos)/atr_dm; ndi=100*wilder(neg)/atr_dm
    den=pdi+ndi; dx=(100*(pdi-ndi).abs()/den).where(den!=0,0).where(den.notna())
    mid=c.rolling(20).mean(); std=c.rolling(20).std(ddof=0)
    spread=h-l; mf=((2*c-h-l)/spread.replace(0,np.nan)).where(spread!=0,0)
    f=pd.DataFrame(index=d.index)
    for n in (10,20,50,150,200):
        f[f'sma{n}']=c.rolling(n).mean(); f[f'sma{n}_slope21_pct']=100*(f[f'sma{n}']/f[f'sma{n}'].shift(21)-1)
    f['atr14']=atr; f['atr_pct']=100*atr/c
    f['adx14']=wilder(dx); f['plus_di14']=pdi; f['minus_di14']=ndi
    f['rvol20']=v/v.shift().rolling(20).mean().replace(0,np.nan)
    f['bb_width20_pct']=400*std/mid
    f['atr_pct_rank126']=f.atr_pct.rolling(126).apply(lambda x:100*(np.sum(x<x[-1])+.5*np.sum(x==x[-1]))/len(x),raw=True)
    f['obv']=(np.sign(c.diff()).fillna(0)*v).cumsum()
    f['cmf20']=(mf*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    for n in (4,7): f[f'nr{n}']=(spread<=spread.rolling(n).min()).where(spread.rolling(n).count()==n)
    f['prior20_high']=h.shift().rolling(20).max(); f['prior20_low']=l.shift().rolling(20).min()
    return f


def features(d, benchmark=None):
    f=feature_series(d); result={k:_f(v) for k,v in f.iloc[-1].items()}
    for n in (4,7):
        val=f[f'nr{n}'].iloc[-1]; result[f'nr{n}']=None if pd.isna(val) else bool(val)
    result['relative_strength']={}
    if benchmark is not None:
        # Exact session intersection; do not silently shorten a requested horizon.
        b=benchmark.close.reindex(d.index)
        for n in (20,63,126,252):
            if len(d)>n and b.iloc[-n-1:].notna().all():
                result['relative_strength'][f'{n}_session_relative_return_pct']=_f(100*((d.close.iloc[-1]/d.close.iloc[-n-1])/(b.iloc[-1]/b.iloc[-n-1])-1))
            else: result['relative_strength'][f'{n}_session_relative_return_pct']=None
    return result


@dataclass(frozen=True)
class Pivot:
    kind: str
    i: int
    at: str
    confirmed_i: int
    confirmed_at: str
    price: float


def confirmed_pivots(d, width=3):
    """Strict symmetric pivots. Knowledge begins at i+width, never at i."""
    hi,lo=d.high.to_numpy(),d.low.to_numpy(); out=[]
    for i in range(width,len(d)-width):
        ph=hi[i]>max(hi[i-width:i]) and hi[i]>max(hi[i+1:i+width+1]); pl=lo[i]<min(lo[i-width:i]) and lo[i]<min(lo[i+1:i+width+1])
        if ph and pl: continue  # intrabar ordering unknown
        if ph or pl: out.append(Pivot('high' if ph else 'low',i,_iso(d.index[i]),i+width,_iso(d.index[i+width]),float(hi[i] if ph else lo[i])))
    return out


def alternate(points):
    """Compress same-type adjacent pivots, retaining the more extreme one.
    Only use on a snapshot at its cutoff; never retroactively timestamp signals.
    """
    out=[]
    for p in points:
        if out and p.kind==out[-1].kind:
            if (p.kind=='high' and p.price>out[-1].price) or (p.kind=='low' and p.price<out[-1].price): out[-1]=p
        else: out.append(p)
    return out


def _rvol(d,i):
    if i<20:return None
    return _ratio(d.volume.iloc[i],d.volume.iloc[i-20:i].mean())


def lifecycle(d, ready_i, trigger, invalidation, bullish, *, min_rvol=1.5, expiry=30):
    """Evaluate after the last structural pivot is known. No same-bar assumed fills."""
    sign=1 if bullish else -1; events=[]; state='ready'; confirmed=None; weak=None
    if sign*(float(d.close.iloc[ready_i])-trigger)>0:
        state='already_beyond_trigger_at_readiness'; weak=ready_i
    for i in range(ready_i+1,len(d)):
        if confirmed is None and i>ready_i+expiry:
            state='expired'; break
        if confirmed is not None and i>confirmed+expiry:
            state='monitoring_window_complete'; break
        c=float(d.close.iloc[i]); prev=float(d.close.iloc[i-1]); vol=_rvol(d,i)
        if sign*(c-invalidation)<0:
            state='invalidated'; events.append({'state':state,'at':_iso(d.index[i]),'price':c});break
        crossed=sign*(c-trigger)>0 and sign*(prev-trigger)<=0
        if confirmed is None and crossed:
            state='confirmed' if vol is not None and vol>=min_rvol else 'price_break_unconfirmed'
            events.append({'state':state,'at':_iso(d.index[i]),'price':c,'relative_volume':vol})
            if state=='confirmed': confirmed=i
            else: weak=i
        elif confirmed is None and weak is not None and sign*(c-trigger)<=0:
            state='ready';weak=None;events.append({'state':'failed_unconfirmed_break','at':_iso(d.index[i]),'price':c})
        elif confirmed is not None and sign*(c-trigger)<=0:
            state='failed_breakout'; events.append({'state':state,'at':_iso(d.index[i]),'price':c}); break
        elif confirmed is not None and i-confirmed>=3 and state=='confirmed':
            state='follow_through';events.append({'state':state,'at':_iso(d.index[i]),'price':c})
    return {'state':state,'events':events,'known_at':_iso(d.index[ready_i]),'confirmed_at':None if confirmed is None else _iso(d.index[confirmed]),'trigger':float(trigger),'invalidation':float(invalidation),'volume_threshold':min_rvol,'expiry_bars':expiry}


def _pattern(d, kind, points, trigger, invalidation, bullish, **extra):
    ready=max(p.confirmed_i for p in points)
    return {'pattern':kind,'direction':'bullish' if bullish else 'bearish','anchors':[asdict(p) for p in points],**lifecycle(d,ready,trigger,invalidation,bullish),**extra}


def patterns(d, width=3):
    p=alternate(confirmed_pivots(d,width)); out=[]
    # Only recent completed structures, with prior trend and minimum size gates.
    for j in range(max(0,len(p)-28),len(p)-2):
        a,b,c=p[j:j+3]; span=c.i-a.i
        if not 8<=span<=90 or a.i<20 or len(d)-1-c.confirmed_i>45:continue
        bullish=a.kind=='low'; kind='double_bottom' if bullish else 'double_top'
        diff=abs(a.price-c.price)/((a.price+c.price)/2)
        if diff>.035:continue
        depth=abs(b.price-(a.price+c.price)/2)/((a.price+c.price)/2)
        if depth<.045:continue
        prior=float(d.close.iloc[a.i-20]); before=float(d.close.iloc[a.i]); prior_move=before/prior-1
        if bullish and prior_move>-.04 or not bullish and prior_move<.04:continue
        out.append(_pattern(d,kind,[a,b,c],b.price,min(a.price,c.price) if bullish else max(a.price,c.price),bullish,depth_pct=100*depth,low_or_high_difference_pct=100*diff,prior20_move_pct=100*prior_move))
    for j in range(max(0,len(p)-24),len(p)-4):
        a,b,c,e,f=p[j:j+5]; bullish=a.kind=='low'
        if a.i<20 or not 15<=f.i-a.i<=125 or len(d)-1-f.confirmed_i>45:continue
        if abs(a.price-f.price)/((a.price+f.price)/2)>.05:continue
        sign=1 if bullish else -1
        if sign*((a.price+f.price)/2-c.price)<=.035*c.price:continue
        if abs(b.price-e.price)/((b.price+e.price)/2)>.06:continue
        prev=d.close.iloc[a.i]/d.close.iloc[a.i-20]-1
        if sign*prev>-.04:continue
        # Conservative horizontal neckline; sloped neckline deliberately not modelled.
        trigger=max(b.price,e.price) if bullish else min(b.price,e.price)
        out.append(_pattern(d,'inverse_head_shoulders' if bullish else 'head_shoulders',[a,b,c,e,f],trigger,c.price,bullish,neckline_method='conservative_horizontal'))
    # Direct prior-range breakout / false breakout / retest, known on close, not hindsight.
    for i in range(max(20,len(d)-30),len(d)):
        prior=d.iloc[i-20:i]; hi=float(prior.high.max());lo=float(prior.low.min());c=float(d.close.iloc[i]); prev=float(d.close.iloc[i-1]);vol=_rvol(d,i)
        for bullish,level,extreme in ((True,hi,float(d.high.iloc[i])),(False,lo,float(d.low.iloc[i]))):
            sign=1 if bullish else -1
            if sign*(extreme-level)>0 and sign*(c-level)<=0:
                out.append({'pattern':'false_break_up' if bullish else 'false_break_down','direction':'bearish' if bullish else 'bullish','state':'observed','known_at':_iso(d.index[i]),'trigger':level,'invalidation':extreme,'relative_volume':vol,'anchors':[],'events':[]})
            elif sign*(c-level)>0 and sign*(prev-level)<=0:
                state='confirmed' if vol is not None and vol>=1.5 else 'price_break_unconfirmed'; events=[{'state':state,'at':_iso(d.index[i])}]
                if state=='confirmed':
                    for k in range(i+1,min(i+11,len(d))):
                        close=float(d.close.iloc[k]); ex=float(d.low.iloc[k] if bullish else d.high.iloc[k])
                        if sign*(close-level)<0:state='failed_breakout';events.append({'state':state,'at':_iso(d.index[k])});break
                        if abs(ex-level)<=.01*level and sign*(close-level)>0:state='retest_held';events.append({'state':state,'at':_iso(d.index[k])});break
                out.append({'pattern':'range_breakout' if bullish else 'range_breakdown','direction':'bullish' if bullish else 'bearish','state':state,'known_at':_iso(d.index[i]),'trigger':level,'invalidation':level,'relative_volume':vol,'anchors':[],'events':events})
    return sorted(out,key=lambda x:x['known_at'],reverse=True)[:20]


def weekly_completed(d):
    """Only fully observed exchange-calendar weeks, including holiday-shortened weeks."""
    import exchange_calendars as xcals
    dates=pd.DatetimeIndex(d.index).tz_convert('America/New_York').tz_localize(None).normalize()
    start=dates[0]-pd.Timedelta(days=7); end=dates[-1]+pd.Timedelta(days=7)
    cal=xcals.get_calendar('XNYS',start=start,end=end)
    # The calendar is clipped to the sessions inside [start,end]. A padded bound on a holiday/weekend lies
    # outside those sessions, so re-querying sessions_in_range(start,end) raised DateOutOfBounds.
    expected=cal.sessions.tz_localize(None)
    groups=dates.to_period('W-FRI'); rows=[]; stamps=[]
    for week in groups.unique():
        mask=groups==week; actual=dates[mask]; exp=expected[expected.to_period('W-FRI')==week]
        if not len(exp) or not actual.equals(exp):continue
        sub=d.iloc[np.flatnonzero(mask)]
        # Require each source timestamp to be the actual completed session close.
        if any(pd.Timestamp(t)!=cal.session_close(day) for t,day in zip(sub.index,exp)):continue
        rows.append({'open':float(sub.open.iloc[0]),'high':float(sub.high.max()),'low':float(sub.low.min()),'close':float(sub.close.iloc[-1]),'volume':float(sub.volume.sum())});stamps.append(sub.index[-1])
    out=pd.DataFrame(rows,index=pd.DatetimeIndex(stamps,tz='UTC'),columns=['open','high','low','close','volume'])
    return out


def stage(d, benchmark=None):
    w=weekly_completed(d)
    if len(w)<36:return {'status':'insufficient_history','completed_weeks':len(w),'stage':None}
    # Reject omitted internal weeks rather than compressing time.
    periods=w.index.tz_convert('America/New_York').tz_localize(None).to_period('W-FRI')
    if not np.all(np.diff(periods.asi8)==1):return {'status':'incomplete_weekly_history','stage':None}
    ma=w.close.rolling(30).mean(); slope=100*(ma.iloc[-1]/ma.iloc[-5]-1); distance=100*(w.close.iloc[-1]/ma.iloc[-1]-1)
    prior=w.iloc[-13:-1]; hi=float(prior.high.max());lo=float(prior.low.min()); c=float(w.close.iloc[-1]); previous_slope=100*(ma.iloc[-2]/ma.iloc[-6]-1)
    if slope>1 and distance>1: label='stage_2_candidate' if c<=hi else 'stage_2_breakout_candidate'
    elif slope<-1 and distance<-1:label='stage_4_candidate' if c>=lo else 'stage_4_breakdown_candidate'
    elif slope>1:label='stage_2_pullback_or_transition'
    elif slope<-1:label='stage_4_rally_or_transition'
    elif previous_slope<-.5:label='stage_1_candidate'
    elif previous_slope>.5:label='stage_3_candidate'
    else:label='stage_1_or_3_unresolved'
    rs=None
    if benchmark is not None:
        b=weekly_completed(benchmark).close.reindex(w.index)
        if b.iloc[-14:].notna().all():rs=_f(100*((w.close.iloc[-1]/w.close.iloc[-14])/(b.iloc[-1]/b.iloc[-14])-1))
    return {'status':'heuristic','stage':label,'completed_weeks':len(w),'as_of_week':_iso(w.index[-1]),'sma30_week':_f(ma.iloc[-1]),'sma30_4week_slope_pct':_f(slope),'distance_pct':_f(distance),'prior12_week_high':hi,'prior12_week_low':lo,'relative_13_week_return_pct':rs,'note':'Weinstein-inspired, not original proprietary stage classification. Flat SMA alone cannot distinguish Stage 1 vs 3.'}


def template(d,benchmark=None, *, rs_percentile=None):
    # 252 prior/including latest completed sessions for a reproducible 52-week proxy.
    if len(d)<252:return {'status':'insufficient_history','observations':len(d),'checks':{}}
    c=d.close;last=float(c.iloc[-1]);ma={n:float(c.iloc[-n:].mean()) for n in (50,150,200)};prev200=float(c.iloc[-221:-21].mean())
    hi=float(d.high.iloc[-252:].max());lo=float(d.low.iloc[-252:].min())
    checks={'above_150_and_200':last>ma[150] and last>ma[200],'150_above_200':ma[150]>ma[200],'200_rising_21_sessions':ma[200]>prev200,'50_above_150_and_200':ma[50]>max(ma[150],ma[200]),'above_50':last>ma[50],'at_least_30pct_above_low':last>=1.3*lo,'within_25pct_of_high':last>=.75*hi,'rs_percentile_at_least_70':None if rs_percentile is None else rs_percentile>=70}
    passed=sum(v is True for v in checks.values()); failed=sum(v is False for v in checks.values())
    status='fails' if failed else 'partial_rs_unverified' if rs_percentile is None else 'passes'
    rs=None
    if benchmark is not None and len(d)>63:
        b=benchmark.close.reindex(d.index)
        if b.iloc[-64:].notna().all():rs=_f(100*((c.iloc[-1]/c.iloc[-64])/(b.iloc[-1]/b.iloc[-64])-1))
    return {'status':status,'checks':checks,'passed':passed,'failed':failed,'unknown':8-passed-failed,'ma':{str(k):v for k,v in ma.items()},'distance_from_52w_low_pct':100*(last/lo-1),'distance_from_52w_high_pct':100*(last/hi-1),'benchmark_relative_63_session_return_pct':rs,'note':'Minervini-inspired 30% low-distance variant. Relative return is not a proprietary RS Rating. No ranking inferred.'}


def vcp(d, width=3):
    p=alternate(confirmed_pivots(d,width));out=[]
    # Look at 2 or 3 complete high-low contractions, with a prior uptrend.
    for count in (3,2):
        for j in range(max(0,len(p)-20),len(p)-2*count+1):
            a=p[j:j+2*count]
            if a[0].kind!='high' or a[0].i<40 or a[-1].i-a[0].i>100 or len(d)-1-a[-1].confirmed_i>35:continue
            if d.close.iloc[a[0].i]/d.close.iloc[a[0].i-40]-1<.12:continue
            pairs=list(zip(a[::2],a[1::2])); contractions=[(h.price-l.price)/h.price for h,l in pairs]
            if any(x<=0 for x in contractions) or not .08<=contractions[0]<=.45 or contractions[-1]>.10:continue
            if any(b>a*.85 for a,b in zip(contractions,contractions[1:])):continue
            if any(y.price<=x.price for x,y in zip(a[1::2],a[3::2])):continue
            highs=[h.price for h,l in pairs]
            if max(highs)/min(highs)>1.12:continue
            vols=[float(d.volume.iloc[h.i:l.i+1].median()) for h,l in pairs]
            if vols[0]<=0 or vols[-1]>vols[0]*.8:continue
            trigger=max(highs);inv=a[-1].price
            out.append(_pattern(d,'vcp_candidate',a,trigger,inv,True,contractions_pct=[100*x for x in contractions],contraction_median_volumes=vols,prior40_return_pct=100*(d.close.iloc[a[0].i]/d.close.iloc[a[0].i-40]-1),note='Heuristic VCP approximation, not an official SEPA signal.'))
    # Preserve distinct anchor sequences but bound output.
    return sorted(out,key=lambda x:(x['known_at'],len(x['anchors'])),reverse=True)[:8]


def overlay_plan(result, provenance):
    """Read-only portable descriptors; user-approved renderer maps to OpenMarket schemas."""
    out=[]
    for row in result[:8]:
        for role in ('trigger','invalidation'):
            out.append({'kind':'horizontal_level','role':role,'price':row[role],'pattern':row['pattern'],'known_at':row['known_at']})
        if row['anchors']:
            out.append({'kind':'pivot_path','pattern':row['pattern'],'anchors':[{'time_ms':int(pd.Timestamp(a['at']).timestamp()*1000),'price':a['price'],'confirmed_at':a['confirmed_at']} for a in row['anchors']]})
    return {'protocol':'stock-chart-overlay/0.1','provenance':provenance,'drawings':out,'requires_identity_session_adjustment_check':True,'auto_apply':False}


def analyze(d, provenance, quality, *, sections=('features','patterns','stage','template','vcp'), benchmark=None, benchmark_provenance=None):
    d=require_ohlcv(d)
    if provenance.get('interval')!='1d' or provenance.get('session')!='regular' or provenance.get('adjustment')!='adjusted':raise DataQualityError('requires adjusted regular-session daily candles')
    if pd.Timestamp(provenance['last_observation'])!=d.index[-1] or pd.Timestamp(provenance['cutoff_at'])<d.index[-1]:raise DataQualityError('provenance cutoff mismatch')
    out={'engine_version':VERSION,'provenance':provenance,'quality':quality,'last_close':float(d.close.iloc[-1]),'status':'experimental','sections':{}}
    if quality.get('status')!='complete' or not quality.get('last_bar_is_latest_expected') or quality.get('observed_bars')!=len(d):
        out['status']='unavailable_incomplete_history';return out
    if benchmark is not None:
        benchmark=require_ohlcv(benchmark)
        if not benchmark_provenance or any(benchmark_provenance.get(k)!=provenance.get(k) for k in ('interval','session','adjustment','currency','cutoff_at')) or benchmark.index[-1]!=d.index[-1]:raise DataQualityError('benchmark provenance mismatch')
        out['benchmark_provenance']=benchmark_provenance
    methods={'features':lambda:features(d,benchmark),'patterns':lambda:patterns(d),'stage':lambda:stage(d,benchmark),'template':lambda:template(d,benchmark),'vcp':lambda:vcp(d)}
    for s in sections:out['sections'][s]=methods[s]()
    rows=out['sections'].get('patterns',[])+out['sections'].get('vcp',[])
    if rows:out['overlay']=overlay_plan(rows,provenance)
    out['limitations']=['Research heuristics, no calibrated probabilities or trade execution.','No point-in-time universe or proprietary RS rank.','No production backtest yet; historical adjusted data may be revised.']
    return out


def intelligence_tools(tool,sym):
    daily={'symbol':sym,'benchmark':sym,'lookback_days':{'type':'integer','minimum':365,'maximum':1250,'default':750}}
    desc={'technical_features':'Daily ADX, DMI, ATR rank, RVOL, OBV, CMF, Bollinger width, NR4/7 and matched benchmark strength.','pattern_scan':'Causal double top/bottom, head and shoulders, range break, false break and retest candidates with lifecycle and chart overlay.','stage_analysis':'Heuristic Weinstein-inspired completed-week stage analysis.','trend_template':'Minervini-inspired trend template; proprietary RS unknown, benchmark relative return separate.','vcp_analysis':'Heuristic volume and volatility contraction candidates with explicit breakout/invalidation.','technical_intelligence':'Combined research snapshot: features, patterns, stage, trend template, VCP and portable read-only chart overlay.'}
    # Normalized chart context is explicitly host-supplied, not a claimed vendor fetch.
    chart_context={'type':'object','additionalProperties':False,'properties':{
        'status':{'enum':['ok','blocked','error','unavailable']},'reason':{'type':'string','maxLength':120},
        'exchange':{'type':'string'},'symbol':{'type':'string'},'interval':{'type':'string'},'currency':{'type':'string'},
        'session':{'type':'string'},'adjustment':{'type':'string'},'replay':{'type':'boolean'},
        'transformations':{'type':'array','maxItems':10,'items':{'type':'string'}},'completed_bar':{'type':'boolean'},
        'bar_end':{'type':'string'},'close':{'type':'number','exclusiveMinimum':0},
        'chart_index':{'type':'integer','minimum':0,'maximum':15},
        'chart_time_by_bar_end_ms':{'type':'object','maxProperties':100,'additionalProperties':{'type':'integer','minimum':100000000000}}},'required':['status']}
    return [tool(name,desc[name]+' Experimental; no trade execution or forecast probability.',{**daily,'chart_context':chart_context} if name=='technical_intelligence' else daily,['symbol']) for name in sorted(NAMES)]


def invoke_intelligence(name,args):
    from .technical import fetch_bars, snapshot
    now=pd.Timestamp.now(tz='UTC'); days=args.get('lookback_days',750)
    d,p,q=fetch_bars(args['symbol'],'1d',days,now=now); b=bp=None; warning=None
    if args.get('benchmark'):
        try:
            candidate,provenance,bq=fetch_bars(args['benchmark'],'1d',days,now=now)
            if bq.get('status')=='complete' and bq.get('last_bar_is_latest_expected') and candidate.index[-1]==d.index[-1]:b,bp=candidate,provenance
            else:warning='Benchmark incomplete or misaligned; relative strength unavailable.'
        except (DataQualityError,ValueError,RuntimeError):warning='Benchmark unavailable; relative strength unavailable.'
    mapping={'technical_features':('features',),'pattern_scan':('patterns',),'stage_analysis':('stage',),'trend_template':('template',),'vcp_analysis':('vcp',),'technical_intelligence':('features','patterns','stage','template','vcp')}
    out=analyze(d,p,q,sections=mapping[name],benchmark=b,benchmark_provenance=bp)
    if warning:out['benchmark_warning']=warning
    if name in ('technical_intelligence','technical_features') and out['status']=='experimental':
        # Reuse the existing canonical calculation on the very same completed bars.
        # No repeated source fetch; standard indicators keep their existing definitions.
        native=snapshot(d,p,q)
        out['standard_technicals']={'data':native['data'],'structure':native['structure'],'calculation_note':native.get('calculation_note')}
    if name=='technical_intelligence' and out['status']=='experimental' and b is not None:
        from .strategy_research import market_timing, MARKET_PROXIES
        if bp.get('symbol') in MARKET_PROXIES and bp.get('asset_type')=='ETF':
            timing=market_timing(b); timing['events']=timing['events'][-20:]
            out['benchmark_market_timing']=timing
    if name=='technical_intelligence' and 'chart_context' in args:
        from .openmarket_bridge import reconcile,drawing_intents
        context=args['chart_context']
        if out['status']!='experimental':
            out['openmarket']={'status':'blocked','reason':'primary_history_incomplete','auto_apply':False}
        else:
            comparison=reconcile(p,context,source_close=out['last_close'])
            out['openmarket']={'comparison':comparison,'source':'host-supplied chart observation, not independently fetched by MCP server'}
            if 'overlay' in out:out['openmarket']['drawing_intents']=drawing_intents(out['overlay'],context,comparison)
    return out
