from __future__ import annotations
import unittest
import numpy as np
from reti_uncertainty import joint_intervals,decide,disagreement_margin,Interval
from reti_sim_epoch_corrected import ACTIONS
from trace_calibration import analyze_trace

class TestUncertainty(unittest.TestCase):
    def test_joint_bounds_ordered(self):
        r=np.random.default_rng(7)
        raw={a:(np.where(r.random(3000)<(.90 if a=='SMS' else .70),1.,np.inf),
                 np.maximum(0.1,r.normal(1.0+i*.2,.10,3000)),np.ones(3000)) for i,a in enumerate(ACTIONS)}
        b=joint_intervals(raw)
        for a in ACTIONS:
            x=b[a]
            self.assertLessEqual(x.qlo,x.qhat)
            self.assertLessEqual(x.qhat,x.qhi)
            self.assertLessEqual(x.elo,x.ehat)
            self.assertLessEqual(x.ehat,x.ehi)
        self.assertIsNotNone(decide(b,.70,'Robust-LCB'))
    def test_strong_certificate(self):
        b={'a':Interval(.96,.98,.9,1.1,.97,1),
           'b':Interval(.82,.88,2.0,2.2,.85,2.1)}
        self.assertEqual(decide(b,.8,'Certified-RETI')[0],'a')
    def test_cannot_ignore_possible_competitor(self):
        b={'a':Interval(.90,.92,1.0,1.1,.91,1.05),
           'b':Interval(.70,.99,.1,.2,.85,.15)}
        x=decide(b,.85,'Certified-RETI')
        self.assertIsNone(x[0]);self.assertEqual(x[1],'uncertain_ranking')
    def test_unfeasible_abstains(self):
        b={'a':Interval(.2,.3,1,2,.25,1.5),'b':Interval(.3,.4,1,2,.35,1.5)}
        self.assertIsNone(decide(b,.85,'Certified-RETI')[0])
    def test_trace_validation(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'trace.csv'; out=Path(d)/'out.csv'
            pd.DataFrame([dict(action='SMS',condition='Normal',recipient_confirmed=1,delivery_latency_ms=125,deadline_ms=300,incremental_energy_mJ=13,measurement_source='pilot')]).to_csv(source,index=False)
            tab=analyze_trace(source,out)
            self.assertEqual(int(tab.n.iloc[0]),1)
            self.assertEqual(float(tab.Q_point.iloc[0]),1.)
            self.assertTrue(out.exists())
    def test_margin_identity(self):
        s={'a':dict(Q=.90,E=2.0),'b':dict(Q=.80,E=1.5)}
        self.assertAlmostEqual(disagreement_margin(s,'a','b'),(.90-.80)/.80-(2-1.5)/1.5)

if __name__=='__main__':unittest.main()
