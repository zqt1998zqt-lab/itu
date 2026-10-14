from pathlib import Path
import csv,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cpp/tools'))
import run_ecnr_v2_paced as p

def fixture(tmp_path):
    engine=dict(input_samples_per_channel=1600,frame_size=160,sample_rate=16000,
        microphones=1,paced_10ms=True,completion_deadline_misses=2,dsp_delay_samples=320)
    (tmp_path/'stream.json').write_text(json.dumps(engine))
    (tmp_path/'stream.after.wav').write_bytes(b'retained output')
    rows=[dict(frame=i,process_ms=.6,release_lateness_ms=.2,completion_late=0) for i in range(10)]
    rows[2].update(release_lateness_ms=11,completion_late=1)
    rows[7].update(release_lateness_ms=9.6,completion_late=1)
    def write():
        with (tmp_path/'stream.frames.csv').open('w',newline='') as f:
            out=csv.DictWriter(f,fieldnames=rows[0].keys());out.writeheader();out.writerows(rows)
    write()
    return engine,rows,write

def test_algorithm_entry_and_completion_kept_separate(tmp_path):
    fixture(tmp_path)
    result=p.trial_metrics(tmp_path,'all repetitions retained')
    assert result['process_frames_over_10ms']==0
    assert result['completion_deadline_misses']==2
    assert result['already_late_before_processing']==1
    assert result['remaining_misses']==1
    assert [int(x['frame']) for x in result['missed_frames']]==[2,7]
    assert result['process_p95_ms']==pytest.approx(.6)
    assert result['process_rtf']==pytest.approx(.06)
    assert result['dsp_delay_ms']==20

def test_count_mismatch_and_invalid_lateness_rejected(tmp_path):
    engine,rows,write=fixture(tmp_path)
    engine['completion_deadline_misses']=1
    (tmp_path/'stream.json').write_text(json.dumps(engine))
    with pytest.raises(ValueError,match='deadline count'):p.trial_metrics(tmp_path,'bad count')
    engine['completion_deadline_misses']=2
    (tmp_path/'stream.json').write_text(json.dumps(engine))
    rows[0]['release_lateness_ms']='nan';write()
    with pytest.raises(ValueError,match='timing unavailable'):p.trial_metrics(tmp_path,'nan')

def test_unpaced_or_missing_frame_rejected(tmp_path):
    engine,rows,write=fixture(tmp_path)
    engine['paced_10ms']=False
    (tmp_path/'stream.json').write_text(json.dumps(engine))
    with pytest.raises(ValueError,match='deadline count'):p.trial_metrics(tmp_path,'offline')
    rows.pop();write()
    with pytest.raises(ValueError,match='missing or unordered'):p.trial_metrics(tmp_path,'missing')

def test_original_p95_rtf_gates_not_replaced_by_ten_ms_only():
    rules={'runtime':{'process_p95_max_ms':1.0,'process_max_ms':10.0,'frames_over_10ms':0,'process_rtf_max':.1}}
    metric={'process_p95_ms':1.1,'process_max_ms':2,'process_frames_over_10ms':0,'process_rtf':.11}
    gates=p.runtime_gates(metric,rules)
    assert [g['metric'] for g in gates if g['status']=='fail']==['process_p95_ms','process_rtf']
    assert [g['threshold'] for g in gates]==[1.0,10.0,0,.1]
