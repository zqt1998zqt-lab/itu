"""Physical power calibration; independent of codec and DSP quality gates."""
import numpy as np
import pytest
from scipy import signal
from server.audio import Audio
from server.graph_engine import engine


@pytest.mark.parametrize('rate',[16000,24000,48000])
def test_stft_sine_dc_nyquist_absolute_and_integrated_power(rate):
    amplitude=.2
    short=engine._frame_size(rate,.008)
    frequency=8*rate/short  # centered in both normal and short FFT grids
    t=np.arange(rate)/rate
    fixtures=[('sine',amplitude*np.cos(2*np.pi*frequency*t),amplitude**2/2),
              ('dc',np.full(rate,amplitude),amplitude**2),
              ('nyquist',amplitude*(-1.)**np.arange(rate),amplitude**2)]
    for name,x,expected in fixtures:
        integrated=[]
        for seconds in [.032,.008]:
            fft=engine._frame_size(rate,seconds)
            f,_,p=engine._stft_power(x,rate,fft,rate//100)
            np.testing.assert_allclose(p.sum(axis=0),expected,rtol=2e-12,atol=1e-15)
            integrated.append(p.sum(axis=0).mean())
            k=int(round(frequency*fft/rate)) if name=='sine' else (0 if name=='dc' else fft//2)
            expected_peak=amplitude**2/3 if name=='sine' else 2*amplitude**2/3
            np.testing.assert_allclose(p[k],expected_peak,rtol=2e-12,atol=1e-15)
            _,_,double=engine._stft_power(x*2,rate,fft,rate//100)
            np.testing.assert_allclose(engine._db(double.sum(axis=0))-engine._db(p.sum(axis=0)),20*np.log10(2),rtol=1e-12)
        assert integrated[0]==pytest.approx(integrated[1],rel=2e-12)


@pytest.mark.parametrize('rate',[16000,24000,48000])
def test_noise_stft_parseval_and_window_length_consistency(rate):
    rng=np.random.default_rng(rate+888)
    x=rng.normal(0,.1,rate*4)
    means=[]
    for seconds in [.032,.008]:
        fft=engine._frame_size(rate,seconds);hop=rate//100 if seconds==.032 else round(rate*.002)
        _,_,p=engine._stft_power(x,rate,fft,hop)
        frames=np.lib.stride_tricks.sliding_window_view(x,fft)[::hop]
        window=signal.get_window('hann',fft,fftbins=True)
        expected=(frames**2 @ (window**2))/np.sum(window**2)
        np.testing.assert_allclose(p.sum(axis=0),expected,rtol=2e-14,atol=1e-15)
        means.append(p.sum(axis=0).mean())
        assert means[-1]==pytest.approx(.01,rel=.025)
    assert means[0]==pytest.approx(means[1],rel=.025)


@pytest.mark.parametrize('rate',[16000,24000,48000])
def test_ten_ms_energy_is_actual_sample_power_including_partial_frame_and_dc(rate):
    hop=rate//100
    x=np.full(hop+17,.2)
    times,p=engine._frame_energy(x,rate)
    np.testing.assert_allclose(p,[.04,.04],atol=1e-15)
    np.testing.assert_allclose(times,[(hop-1)/2/rate,(hop+8)/rate])
    assert engine._db(p)[0]==pytest.approx(10*np.log10(.04),abs=1e-12)
    sine=.2*np.cos(2*np.pi*1000*np.arange(rate)/rate)
    _,p=engine._frame_energy(sine,rate)
    np.testing.assert_allclose(p,.02,atol=1e-14)
    assert engine._db(p)[0]==pytest.approx(-16.989700043360187,abs=1e-11)
    _,louder=engine._frame_energy(sine*2,rate)
    np.testing.assert_allclose(engine._db(louder)-engine._db(p),6.020599913279624,atol=1e-12)


def test_ltas_density_integrates_to_sine_noise_power_and_explicitly_detrends_dc():
    rate=16000;t=np.arange(rate*4)/rate
    sine=.2*np.cos(2*np.pi*1000*t)
    f,psd=engine._ltas_density(sine,rate);df=f[1]-f[0]
    assert psd.sum()*df==pytest.approx(.02,rel=1e-12)
    _,louder=engine._ltas_density(2*sine,rate)
    assert 10*np.log10(louder.max()/psd.max())==pytest.approx(6.020599913279624,abs=1e-12)
    rng=np.random.default_rng(99);noise=rng.normal(0,.1,len(t))
    f,psd=engine._ltas_density(noise,rate)
    assert psd.sum()*(f[1]-f[0])==pytest.approx(.01,rel=.025)
    _,dc=engine._ltas_density(np.full(len(t),.2),rate)
    assert dc.sum()==pytest.approx(0,abs=1e-25)


def test_figure_axes_and_report_state_the_calibrated_units_without_altering_audio(monkeypatch):
    rate=16000;x=.2*np.cos(2*np.pi*1000*np.arange(rate)/rate);original=x.copy()
    saved=[]
    def capture(figure):
        saved.append(figure);return b'physical-units-test'
    monkeypatch.setattr(engine,'_save_png',capture)
    before=Audio('before','before',x,rate,1,0,'WAV','a'*64,x.nbytes)
    after=Audio('after','after',x*2,rate,1,0,'WAV','b'*64,x.nbytes)
    report,figures=engine.analyze_graphics(before,after,engine.GraphOptions(sample_rate=rate,align=False))
    assert len(figures)==8 and len(saved)==8
    scope=report['figure_scope']
    assert scope['stft_power_unit']=='FS^2/bin' and scope['ltas_unit']=='dBFS/Hz'
    assert scope['power_calibration']=='one_sided_mean_square_v2'
    energy=saved[0].axes[-1]
    np.testing.assert_allclose(energy.lines[0].get_ydata(),10*np.log10(.02),atol=1e-11)
    np.testing.assert_allclose(energy.lines[1].get_ydata(),10*np.log10(.08),atol=1e-11)
    assert any(axis.get_ylabel()=='PSD (dBFS/Hz)' for fig in saved for axis in fig.axes)
    np.testing.assert_array_equal(x,original)
