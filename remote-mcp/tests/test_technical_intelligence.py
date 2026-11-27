import json
import unittest
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from stock_research import technical_intelligence as ti
from stock_research.core import DataQualityError


def frame(prices, volume=None, start='2023-01-03'):
    n=len(prices); cal=xcals.get_calendar('XNYS',start='2022-12-01',end='2027-01-01')
    sessions=cal.sessions_in_range(start,'2026-12-31')[:n]
    idx=pd.DatetimeIndex([cal.session_close(s) for s in sessions]); c=np.asarray(prices,dtype=float)
    return pd.DataFrame({'open':c,'high':c*1.005,'low':c*.995,'close':c,'volume':np.ones(n)*1000 if volume is None else volume},index=idx)


def context(d):
    p={'symbol':'TEST','interval':'1d','session':'regular','adjustment':'adjusted','currency':'USD','last_observation':d.index[-1].isoformat(),'cutoff_at':d.index[-1].isoformat()}
    q={'status':'complete','last_bar_is_latest_expected':True,'observed_bars':len(d)}
    return p,q


def path(anchors):
    n=anchors[-1][0]+1;return np.interp(np.arange(n),[i for i,v in anchors],[v for i,v in anchors])


class TechnicalIntelligenceTests(unittest.TestCase):
    def test_features_rvol_excludes_current(self):
        d=frame(np.linspace(100,160,300));d.iloc[-1,d.columns.get_loc('volume')]=2000
        out=ti.features(d); self.assertAlmostEqual(out['rvol20'],2); self.assertGreater(out['adx14'],90);self.assertGreater(out['plus_di14'],out['minus_di14']);self.assertTrue(0<=out['atr_pct_rank126']<=100)
    def test_flat_volume_unknown(self):
        d=frame(np.ones(35)*100,np.zeros(35)); f=ti.features(d)
        self.assertIsNone(f['rvol20']);self.assertIsNone(f['cmf20']);self.assertIsNone(f['sma200']); json.dumps(f,allow_nan=False)
    def test_feature_prefix_causal(self):
        d=frame(np.linspace(100,160,300)+2*np.sin(np.arange(300)))
        pd.testing.assert_frame_equal(ti.feature_series(d.iloc[:260]),ti.feature_series(d).iloc[:260])
    def test_pivot_confirmation(self):
        d=frame([10,11,12,15,12,11,10,9,10,11,12]);p=ti.confirmed_pivots(d,3)
        self.assertEqual(p[0].i,3);self.assertEqual(p[0].confirmed_i,6);self.assertFalse(ti.confirmed_pivots(d.iloc[:6],3));self.assertEqual(ti.confirmed_pivots(d.iloc[:7],3)[0],p[0])
    def test_lifecycle_breakout_failure(self):
        d=frame([100]*25+[112,114,115,116,109],np.array([1000]*25+[2000,1000,1000,1000,1000]))
        result=ti.lifecycle(d,24,110,95,True)
        self.assertEqual(result['state'],'failed_breakout');self.assertEqual(result['confirmed_at'],d.index[25].isoformat());self.assertEqual(ti.lifecycle(d.iloc[:25],24,110,95,True)['state'],'ready');self.assertEqual(ti.lifecycle(d.iloc[:29],24,110,95,True)['state'],'follow_through')
    def test_double_bottom_ready_and_confirmed(self):
        d=frame(path([(0,145),(25,100),(35,112),(45,101),(50,105),(54,111),(55,115),(56,116)]));d.iloc[55,d.columns.get_loc('volume')]=3000
        before=[p for p in ti.patterns(d.iloc[:48]) if p['pattern']=='double_bottom'];self.assertEqual(before,[])
        ready=[p for p in ti.patterns(d.iloc[:50]) if p['pattern']=='double_bottom'];self.assertEqual(ready[0]['state'],'ready')
        after=[p for p in ti.patterns(d) if p['pattern']=='double_bottom'];self.assertEqual(after[0]['state'],'confirmed');self.assertEqual(after[0]['confirmed_at'],d.index[55].isoformat())
    def test_stage_completed_week_only(self):
        d=frame(np.linspace(100,180,310));w=ti.weekly_completed(d); self.assertTrue(len(w)>36); self.assertIn('stage_2',ti.stage(d)['stage']); self.assertIn('stage_4',ti.stage(frame(np.linspace(180,100,310)))['stage'])
        first_week=w.index[0]; i=d.index.get_loc(first_week);self.assertEqual(len(ti.weekly_completed(d.iloc[:i])),0);self.assertEqual(len(ti.weekly_completed(d.iloc[:i+1])),1)
    def test_weekly_completed_when_padded_bounds_are_not_sessions(self):
        # Regression: the +/-7 day padding was passed back to sessions_in_range; when a padded bound fell on a
        # market holiday the calendar was clipped inside it and the query raised DateOutOfBounds.
        cal=xcals.get_calendar('XNYS',start='2024-12-01',end='2027-01-01');week=pd.Timedelta(days=7)
        for first,last,first_week,last_week in (('2025-01-08','2025-04-02','2025-01-17','2025-03-28'),   # first-7d: New Year's Day
                                                ('2026-06-08','2026-11-19','2026-06-12','2026-11-13'),   # last+7d: Thanksgiving
                                                ('2025-01-08','2026-11-19','2025-01-17','2026-11-13')):  # both
            n=len(cal.sessions_in_range(first,last));d=frame(np.linspace(100,120,n),start=first)
            self.assertEqual(d.index[-1],cal.session_close(last))
            self.assertFalse(cal.is_session(pd.Timestamp(first)-week) and cal.is_session(pd.Timestamp(last)+week))
            w=ti.weekly_completed(d)
            self.assertEqual(w.index[0],cal.session_close(first_week));self.assertEqual(w.index[-1],cal.session_close(last_week))
            self.assertEqual(len(w),len(pd.period_range(first_week,last_week,freq='W-FRI')))
    def test_stage_missing_week_rejected(self):
        d=frame(np.linspace(100,180,310));w=ti.weekly_completed(d);i=d.index.get_loc(w.index[20]);d=d.drop(d.index[i]);self.assertEqual(ti.stage(d)['status'],'incomplete_weekly_history')
    def test_trend_template_missing_rs(self):
        d=frame(np.linspace(100,180,300));r=ti.template(d);self.assertEqual(r['status'],'partial_rs_unverified');self.assertEqual(r['unknown'],1);self.assertIsNone(r['checks']['rs_percentile_at_least_70']);self.assertEqual(ti.template(d.iloc[:240])['status'],'insufficient_history')
    def test_benchmark_no_fake_rank(self):
        d=frame(np.linspace(100,180,300));b=frame(np.linspace(100,150,300));b=b.drop(b.index[-10]);self.assertIsNone(ti.features(d,b)['relative_strength']['20_session_relative_return_pct'])
    def test_vcp(self):
        c=path([(0,70),(45,115),(55,93),(65,115),(75,104),(85,115),(95,110),(100,113),(104,117),(106,118)])
        v=np.ones(len(c))*1000;v[45:56]=2000;v[65:76]=1000;v[85:96]=500;v[103]=3000
        d=frame(c,v);r=ti.vcp(d);self.assertTrue(any(len(p['anchors'])==6 for p in r));self.assertTrue(any(p['state']=='follow_through' and p['confirmed_at']==d.index[103].isoformat() for p in r));self.assertFalse(any(len(p['anchors'])==6 for p in ti.vcp(d.iloc[:98])))
    def test_weak_break_cannot_be_confirmed_by_later_volume(self):
        d=frame([100]*25+[112,114,115,116],np.array([1000]*25+[1000,3000,1000,1000]))
        r=ti.lifecycle(d,24,110,95,True)
        self.assertEqual(r['state'],'price_break_unconfirmed');self.assertIsNone(r['confirmed_at'])
    def test_already_beyond_trigger_when_pivot_known(self):
        d=frame([100]*23+[111,112,115,116],np.array([1000]*25+[3000,1000]))
        r=ti.lifecycle(d,24,110,95,True)
        self.assertEqual(r['state'],'already_beyond_trigger_at_readiness');self.assertIsNone(r['confirmed_at'])
    def test_bad_quality_fails_closed(self):
        d=frame(np.linspace(100,180,300));p,q=context(d);q['status']='degraded'; self.assertEqual(ti.analyze(d,p,q)['status'],'unavailable_incomplete_history')
        p['interval']='1h';self.assertRaises(DataQualityError,ti.analyze,d,p,q)
    def test_tool_schemas(self):
        from jsonschema import Draft202012Validator
        def tool(name,description,properties,required):
            return {'name':name,'inputSchema':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}
        specs=ti.intelligence_tools(tool,{'type':'string'})
        self.assertEqual({x['name'] for x in specs},ti.NAMES)
        for spec in specs:
            Draft202012Validator.check_schema(spec['inputSchema'])
        combined=next(x['inputSchema'] for x in specs if x['name']=='technical_intelligence')
        self.assertTrue(Draft202012Validator(combined).is_valid({'symbol':'QQQ','chart_context':{'status':'blocked','reason':'no_tab'}}))
        self.assertFalse(Draft202012Validator(combined).is_valid({'symbol':'QQQ','chart_context':{'status':'ok','secret':'not-allowed'}}))
    def test_json_and_overlay(self):
        d=frame(np.linspace(100,180,300));p,q=context(d);r=ti.analyze(d,p,q);json.dumps(r,allow_nan=False)
        plan=ti.overlay_plan([{'pattern':'sample','trigger':110,'invalidation':100,'known_at':d.index[-1].isoformat(),'anchors':[]}],p);self.assertFalse(plan['auto_apply']);self.assertEqual(len(plan['drawings']),2)


class CombinedToolIntegrationTests(unittest.TestCase):
    def test_shared_fetch_standard_indicators_and_benchmark_timing(self):
        from unittest.mock import patch
        from types import ModuleType
        import sys
        from stock_research import technical_intelligence as ti
        d=frame(np.linspace(100,150,300)); p,q=context(d); p.update({'symbol':'MSFT','asset_type':'EQUITY','currency':'USD'})
        b=frame(np.linspace(100,130,300)); bp={**p,'symbol':'SPY','asset_type':'ETF'}
        stub=ModuleType('stock_research.technical')
        with patch.dict(sys.modules,{'stock_research.technical':stub}):
            with patch.object(stub,'fetch_bars',create=True,side_effect=[(d,p,q),(b,bp,q)]) as fetch, patch.object(stub,'snapshot',create=True,return_value={'data':{'close':150},'structure':{},'calculation_note':'canonical'}) as snap:
                r=ti.invoke_intelligence('technical_intelligence',{'symbol':'MSFT','benchmark':'SPY'})
                self.assertEqual(fetch.call_count,2);self.assertEqual(snap.call_count,1)
                self.assertEqual(r['standard_technicals']['data']['close'],150);self.assertIn('benchmark_market_timing',r)

if __name__=='__main__':unittest.main()
