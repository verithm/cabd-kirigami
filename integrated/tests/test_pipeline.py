"""Small deterministic integration tests; no private data or model training required."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

_TEMP = tempfile.TemporaryDirectory()
os.environ['CABD_INTEGRATED_RUNS'] = str(Path(_TEMP.name) / 'private-runs')
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
for folder in [BASE, BASE/'preprocessing', BASE/'forward', BASE/'inverse', ROOT/'src']:
    sys.path.insert(0, str(folder))
import numpy as np
from runtime_paths import RUNS, reference, resolve_reference
from preprocess_energy_targets import first_energy_well, prepare
from common import FEATURES, TARGETS
from shared import normalized, to_phys, bound_mask
from cabd.geometry import params as G
from goal_range_benchmark import error_matrix, score
from evaluate_no_barrier import SOURCE, DEST, reevaluate


def csv_text(rows):
    stream = io.StringIO();writer = csv.DictWriter(stream, list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue()

class CommonPipelineTests(unittest.TestCase):
    def test_external_runs_references_and_legacy_compatibility(self):
        p = RUNS/'unified-inverse/runs/a.npz'
        self.assertEqual(resolve_reference(reference(p)), p)
        self.assertEqual(resolve_reference('output/unified-inverse/runs/a.npz'), p)
        self.assertEqual(resolve_reference('output'+chr(92)+'unified-inverse'+chr(92)+'runs'+chr(92)+'a.npz'), p)

    def test_initial_protocol_and_inverse_status(self):
        status=json.loads((BASE/'configs/model_status.json').read_text())
        self.assertIsNone(status['inverse']['selected_model'])
        self.assertFalse(status['integrated_ansys_validation'])
        self.assertTrue(status['no_barrier_experiment']['barrier_still_in_generator_input'])

    def test_energy_peak_followed_by_valley(self):
        self.assertEqual(first_energy_well(np.array([0,1,2,1,.5,1])),('bistable',(2,4)))
        self.assertEqual(first_energy_well(np.array([0,1,2])),('monostable',None))
        self.assertEqual(first_energy_well(np.array([2,1,0])),('unknown',None))

    def test_plateau_extrema(self):
        self.assertEqual(first_energy_well(np.array([0,1,2,2,1,0,0,1])),('bistable',(2,5)))

    def test_energy_extraction_quality_gates_and_landmarks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'synthetic.zip';index=root/'index.csv'
            common=dict(state='partial',role='train',**{k:str(G.REF[k]) for k in FEATURES},mesh_rule='test',archive=archive.name)
            rows=[]
            for i,status,complete in [(1,'ok','1'),(2,'not_converged','1'),(3,'ok','0'),(4,'ok','1')]:
                rows.append(dict(common,geom_id=f'TR{i:04d}',status=status,force_complete=complete,curve_file=f'TR{i:04d}/partial_curve.csv'))
            index.write_text(csv_text(rows))
            with ZipFile(archive,'w') as z:
                for gid,energy in [('TR0001',[0,2,1,.5,1.5]),('TR0004',[0,1,float('nan'),2,3])]:
                    z.writestr(f'{gid}/partial_curve.csv',csv_text([dict(set=i+1,U_total_mJ=e,u_applied_mm=u) for i,(e,u) in enumerate(zip(energy,[0,3,6,9,15]))]))
                nodes=[dict(name=n,landmark=i+1,x0_mm=x,z0_mm=z) for i,(n,x,z) in enumerate([('A_upper',0,1),('B_upper',1,1),('A_lower',0,-1),('B_lower',1,-1)])]
                disp=[dict(set=4,landmark=i+1,ux_mm=0,uz_mm=dz) for i,dz in enumerate([0,-1,0,1])]
                z.writestr('TR0001/partial_landmark_nodes.csv',csv_text(nodes));z.writestr('TR0001/partial_landmark_disp.csv',csv_text(disp))
            result=list(prepare({archive.name:archive},index))
            self.assertEqual(len(result),2)
            self.assertEqual(result[0]['stability'],'bistable');self.assertAlmostEqual(result[0]['reverse_barrier_mJ'],1.5)
            self.assertAlmostEqual(result[0]['deformation_applied_mm'],9);self.assertAlmostEqual(result[0]['stable_angle_deg'],-45)
            self.assertEqual(result[1]['stability'],'unknown');self.assertIsNone(result[1]['stable_angle_deg'])

    def test_geometry_units_roundtrip_and_bounds(self):
        x=np.array([G.REF[k] for k in FEATURES]);np.testing.assert_allclose(to_phys(normalized(x)),x)
        self.assertTrue(bound_mask(normalized(x)[None])[0]);self.assertFalse(bound_mask(np.full((1,8),1.1))[0])

    def test_physical_energy_error_not_log_error(self):
        target=np.array([60.,12.,np.log10(.5)])
        pred=np.array([[62.,12.25,np.log10(.53)]])
        np.testing.assert_allclose(error_matrix(pred,target),[[1,1,1]],atol=1e-12)

    def test_barrier_removal_preserves_pool_and_probability_gate(self):
        strategy='2_interpolation';method='jw_cvae';gid='test_goal'
        (SOURCE/'goals').mkdir(parents=True,exist_ok=True);folder=SOURCE/'runs'/strategy/method;folder.mkdir(parents=True,exist_ok=True)
        target=np.array([60.,12.,np.log10(.5)])
        (SOURCE/'goals'/f'{strategy}.csv').write_text(csv_text([dict(goal_id=gid,**dict(zip(TARGETS,target)))]))
        u=np.tile(normalized([G.REF[k] for k in FEATURES]),(3,1));y=np.array([[60.,12.,np.log10(.6)],[60.,12.,np.log10(.51)],[60.,12.,np.log10(.5)]])
        np.savez(folder/f'{gid}.npz',u=u,predicted=y,p_bi_mono=np.array([.95,.95,.89],dtype=np.float32),geometry_checked=np.array([-1,1,-1],dtype=np.int8))
        (folder/f'{gid}.json').write_text(json.dumps(dict(goal=dict(zip(TARGETS,target)),goal_id=gid,success=True,generated_pass_count=1)))
        reevaluate(strategy,method,0,1)
        result=json.loads((DEST/'runs'/strategy/method/f'{gid}.json').read_text())
        self.assertEqual(result['generated_pass_count'],2);self.assertTrue(result['success'])
        self.assertAlmostEqual(result['selected']['predicted_barrier_mJ'],.6)

    def test_cli_help_has_no_data_dependency(self):
        r=subprocess.run([sys.executable,str(BASE/'cli.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('without-barrier',r.stdout)

if __name__=='__main__':unittest.main()
