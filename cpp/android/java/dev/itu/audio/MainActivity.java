package dev.itu.audio;
import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AudioEffect;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import android.widget.EditText;
import android.view.View;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.json.JSONObject;

/** Device integration demo. No recording until the user presses the capture button. */
public final class MainActivity extends Activity {
    private TextView status;
    private volatile boolean running=false, stopCapture=false;
    private Thread worker;
    private Spinner microphoneChoice, rateChoice, noiseChoice;
    private int selectedRate=16000,selectedNoiseBackend=0;
    private EditText beamDelays;
    private int selectedMicrophones=1;
    private float[] selectedDelays={0};
    private static final int RATE=16000, FRAME=160, CAPTURE_FRAMES=1000;
    private void show(String text) {Log.i("ITU_AUDIO",text); runOnUiThread(()->status.setText(text));}
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout layout=new LinearLayout(this); layout.setOrientation(LinearLayout.VERTICAL); layout.setPadding(32,48,32,32);
        TextView title=new TextView(this); title.setText("ITU · Android 音频工程"); title.setTextSize(24); layout.addView(title);
        TextView details=new TextView(this); details.setText("AEC3 · NS · AGC2"); details.setTextSize(16); layout.addView(details);
        TextView rateLabel=new TextView(this); rateLabel.setText("采样率"); layout.addView(rateLabel);
        rateChoice=new Spinner(this); rateChoice.setAdapter(new ArrayAdapter<String>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"16 kHz","24 kHz","48 kHz"})); layout.addView(rateChoice);
        TextView nsLabel=new TextView(this); nsLabel.setText("降噪器"); layout.addView(nsLabel);
        noiseChoice=new Spinner(this); noiseChoice.setAdapter(new ArrayAdapter<String>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"WebRTC NS","RNNoise","SpeexDSP"})); layout.addView(noiseChoice);
        TextView microphonesLabel=new TextView(this); microphonesLabel.setText("麦克风路数"); layout.addView(microphonesLabel);
        microphoneChoice=new Spinner(this);
        microphoneChoice.setAdapter(new ArrayAdapter<String>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"1 路","2 路","4 路"})); layout.addView(microphoneChoice);
        TextView delaysLabel=new TextView(this); delaysLabel.setText("各路补偿延时 / 样本"); layout.addView(delaysLabel);
        beamDelays=new EditText(this); beamDelays.setSingleLine(true); beamDelays.setText("0"); layout.addView(beamDelays);
        microphoneChoice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            public void onNothingSelected(AdapterView<?> parent) {}
            public void onItemSelected(AdapterView<?> parent,View view,int position,long id) {
                beamDelays.setText(position==0 ? "0" : position==1 ? "0,0" : "0,0,0,0");
            }
        });
        Button test=new Button(this); test.setText("运行 JNI 自检（无录音）"); test.setOnClickListener(v->start(false)); layout.addView(test);
        Button record=new Button(this); record.setText("采集并处理 10 秒音频"); record.setOnClickListener(v->{
            if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED) requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},7);
            else start(true);
        }); layout.addView(record);
        Button stop=new Button(this); stop.setText("停止采集"); stop.setOnClickListener(v->stopCapture=true); layout.addView(stop);
        status=new TextView(this); status.setText("就绪。自检使用明确标注的合成数据；采集使用设备麦克风。"); status.setTextSize(16); layout.addView(status);
        setContentView(layout);
        if(getIntent().getBooleanExtra("self_test",false)) start(false);
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] grants) {
        super.onRequestPermissionsResult(request,permissions,grants);
        if(request==7 && grants.length>0 && grants[0]==PackageManager.PERMISSION_GRANTED) start(true);
        else show("没有获得麦克风权限。");
    }
    private void start(boolean capture) {
        if(running) {show("任务正在运行。"); return;}
        if(capture) {
            selectedRate=new int[]{16000,24000,48000}[rateChoice.getSelectedItemPosition()];
            selectedNoiseBackend=noiseChoice.getSelectedItemPosition();
            selectedMicrophones=new int[]{1,2,4}[microphoneChoice.getSelectedItemPosition()];
            String[] parts=beamDelays.getText().toString().split(",",-1);
            if(parts.length!=selectedMicrophones) {show("补偿延时数量必须与麦克风路数一致。"); return;}
            selectedDelays=new float[selectedMicrophones];
            try {
                for(int ch=0;ch<parts.length;ch++) {
                    selectedDelays[ch]=Float.parseFloat(parts[ch].trim());
                    if(!Float.isFinite(selectedDelays[ch]) || selectedDelays[ch]<0 || selectedDelays[ch]>4096)
                        throw new IllegalArgumentException();
                }
            } catch(IllegalArgumentException e) {show("补偿延时须为 0 至 4096 的有限数值。"); return;}
        }
        running=true; stopCapture=false;
        worker=new Thread(()->{
            try {if(capture) capture(); else selfTest();} catch(Exception e) {Log.e("ITU_AUDIO","task failed",e); show("失败："+e);}
            finally {running=false;}
        },"itu-audio-stream"); worker.start();
    }
    private void save(String name,JSONObject object) throws Exception {
        try(FileOutputStream f=new FileOutputStream(new File(getFilesDir(),name))) {f.write(object.toString(2).getBytes(StandardCharsets.UTF_8));}
    }
    private void selfTest() throws Exception {
        show("正在检查 JNI 缓冲区和完整处理链……");
        ByteBuffer mic=NativeEngine.floats(FRAME), ref=NativeEngine.floats(FRAME), out=NativeEngine.floats(FRAME), stats=NativeEngine.floats(16);
        int checks=0;
        try(NativeEngine bypass=new NativeEngine(RATE,FRAME,1,0,2,0,null)) {
            for(int i=0;i<FRAME;i++) mic.putFloat(i*4,i*.001f);
            if(bypass.process(mic,null,out,stats)!=0) throw new IllegalStateException("bypass status");
            for(int i=0;i<FRAME;i++) if(mic.getFloat(i*4)!=out.getFloat(i*4)) throw new IllegalStateException("bypass mismatch"); checks++;
            if(bypass.process(ByteBuffer.allocate(FRAME*4),null,out,stats)!=-2) throw new IllegalStateException("heap buffer accepted"); checks++;
            if(bypass.process(NativeEngine.floats(FRAME-1),null,out,stats)!=-2) throw new IllegalStateException("short buffer accepted"); checks++;
            if(bypass.process(mic,null,out.asReadOnlyBuffer().order(ByteOrder.nativeOrder()),stats)!=-2) throw new IllegalStateException("readonly output accepted"); checks++;
            ByteBuffer base=NativeEngine.floats(FRAME+1); base.position(1); base.limit(FRAME*4+1);
            if(bypass.process(base.slice().order(ByteOrder.nativeOrder()),null,out,stats)!=-2) throw new IllegalStateException("misaligned buffer accepted"); checks++;
            mic.putFloat(0,Float.NaN); if(bypass.process(mic,null,out,stats)!=-3) throw new IllegalStateException("NaN accepted"); checks++;
            mic.putFloat(0,0); if(bypass.process(mic,null,out,stats)!=0 || stats.getLong(0)!=1) throw new IllegalStateException("invalid frame advanced state"); checks++;
            bypass.reset(); if(bypass.process(mic,null,out,stats)!=0 || stats.getLong(0)!=0) throw new IllegalStateException("reset state"); checks++;
        }
        double[] timings=new double[1000]; int seed=193, linearFrames=0; long memory;
        ByteBuffer aecTap=NativeEngine.floats(FRAME),bfTap=NativeEngine.floats(FRAME),linearTap=NativeEngine.floats(FRAME);
        try(NativeEngine engine=new NativeEngine(RATE,FRAME,1,91,2,0,new float[]{0})) {
            if(engine.process(mic,null,out,stats)!=-2) throw new IllegalStateException("AEC missing reference accepted"); checks++;
            memory=engine.workingMemoryBytes();
            for(int k=0;k<1200;k++) {
                for(int i=0;i<FRAME;i++) {seed=seed*1664525+1013904223; float s=((seed>>>16)-32768)/32768f*.2f; ref.putFloat(i*4,s); mic.putFloat(i*4,.6f*s);}
                long t=System.nanoTime(); int result=engine.processWithTaps(mic,ref,out,stats,aecTap,bfTap,linearTap); long elapsed=System.nanoTime()-t;
                if(result!=0) throw new IllegalStateException("full chain status "+result);
                if(k>=200) {timings[k-200]=elapsed/1e6; if((stats.getInt(8)&2)!=0) linearFrames++;}
                for(int i=0;i<FRAME;i++) if(!Float.isFinite(out.getFloat(i*4))) throw new IllegalStateException("non-finite output");
            }
            if(stats.getInt(36)!=480 || stats.getInt(52)!=384 || stats.getInt(56)!=320 || linearFrames!=1000 || (stats.getInt(40)&2)==0) throw new IllegalStateException("latency/linear AEC/stats"); checks++;
        }
        try(NativeEngine vehicle=new NativeEngine(RATE,FRAME,1,NativeEngine.NS|NativeEngine.AGC|NativeEngine.LIMITER,0,0,null,NativeEngine.PROFILE_VEHICLE_STREAM)) {
            for(int k=0;k<20;k++) {
                if(vehicle.process(mic,null,out,stats)!=0 || stats.getLong(0)!=k || stats.getInt(36)!=352)
                    throw new IllegalStateException("vehicle profile without playback reference");
            }
            checks++;
        }
        org.json.JSONArray rateChecks=new org.json.JSONArray();
        for(int rate:new int[]{16000,24000,48000}) for(int channels:new int[]{1,2,4}) for(int backend:new int[]{0,1,2}) {
            if(!NativeEngine.backendAvailable(backend)) continue;
            int frame=rate/100;
            ByteBuffer input=NativeEngine.floats(frame*channels), playback=NativeEngine.floats(frame);
            ByteBuffer output=NativeEngine.floats(frame), info=NativeEngine.floats(16);
            ByteBuffer aec=NativeEngine.floats(frame*channels), bf=NativeEngine.floats(frame), linear=NativeEngine.floats(160*channels);
            try(NativeEngine engine=new NativeEngine(rate,frame,channels,91,0,0,new float[channels],NativeEngine.PROFILE_VEHICLE_STREAM,backend)) {
                for(int k=0;k<100;k++) {
                    for(int i=0;i<frame;i++) {
                        float value=(float)(.03*Math.sin(2*Math.PI*997*(k*frame+i)/rate));
                        playback.putFloat(i*4,0);
                        for(int ch=0;ch<channels;ch++) input.putFloat((i*channels+ch)*4,value);
                    }
                    if(engine.processWithTaps(input,playback,output,info,aec,bf,linear)!=0 || info.getLong(0)!=k || (info.getInt(8)&2)==0)
                        throw new IllegalStateException("multirate JNI/tap contract");
                    for(int i=0;i<frame;i++) if(!Float.isFinite(output.getFloat(i*4))) throw new IllegalStateException("multirate non-finite");
                }
                if(rate!=16000 && engine.processWithTaps(input,playback,output,info,aec,bf,NativeEngine.floats(159*channels))!=-2)
                    throw new IllegalStateException("short 16k linear tap accepted");
                JSONObject row=new JSONObject(); row.put("sample_rate",rate); row.put("frame_size",frame);
                row.put("microphones",channels); row.put("ns_backend",backend); row.put("dsp_delay_samples",info.getInt(36));
                row.put("linear_sample_rate",16000); row.put("passed",true); rateChecks.put(row); checks++;
            }
        }
        Arrays.sort(timings);
        int deadlineMisses=0; for(double t:timings) if(t>10) deadlineMisses++;
        JSONObject report=new JSONObject(); report.put("schema_version",2); report.put("webrtc_revision",NativeEngine.revision()); report.put("passed",true); report.put("checks",checks);
        report.put("run_id",getIntent().getStringExtra("run_id")); report.put("completed_at_epoch_ms",System.currentTimeMillis());
        report.put("input_kind","deterministic synthetic; microphone not opened"); report.put("timing_scope","Java validation + JNI + C++ process, unpaced");
        report.put("frames",1000); report.put("p50_ms",timings[499]); report.put("p95_ms",timings[949]); report.put("max_ms",timings[999]); report.put("wrapper_memory_bytes",memory); report.put("apm_memory_bytes",JSONObject.NULL); report.put("dtd_status_available",false);
        report.put("deadline_misses",deadlineMisses); report.put("dsp_delay_samples",480);
        report.put("multirate_interface_checks",rateChecks); report.put("multirate_quality_accepted",false);
        save("selftest.json",report); show("JNI 自检通过："+checks+" 项\nP95 "+timings[949]+" ms / 10 ms 帧\n报告：selftest.json");
    }
    private static boolean disable(AudioEffect effect) {return effect==null || effect.setEnabled(false)==AudioEffect.SUCCESS;}
    private void capture() throws Exception {
        final int RATE=selectedRate,FRAME=RATE/100,noiseBackend=selectedNoiseBackend;
        final int microphones=selectedMicrophones, frameSamples=FRAME*microphones;
        final float[] delays=selectedDelays.clone();
        File oldReport=new File(getFilesDir(),"capture.json");
        if(oldReport.exists() && !oldReport.delete()) throw new IllegalStateException("cannot invalidate previous capture report");
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
        AudioManager manager=(AudioManager)getSystemService(AUDIO_SERVICE);
        boolean unprocessed="true".equals(manager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED));
        int source=unprocessed ? MediaRecorder.AudioSource.UNPROCESSED : MediaRecorder.AudioSource.VOICE_RECOGNITION;
        int minimum=AudioRecord.getMinBufferSize(RATE,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
        if(minimum<=0) throw new IllegalStateException(RATE+" Hz capture unsupported: "+minimum);
        AudioFormat.Builder format=new AudioFormat.Builder().setSampleRate(RATE).setEncoding(AudioFormat.ENCODING_PCM_16BIT);
        if(microphones==1) format.setChannelMask(AudioFormat.CHANNEL_IN_MONO);
        else format.setChannelIndexMask((1<<microphones)-1);
        AudioRecord recorder=new AudioRecord.Builder().setAudioSource(source).setAudioFormat(format.build())
            .setBufferSizeInBytes(Math.max(minimum*microphones*2,frameSamples*8)).build();
        AcousticEchoCanceler aec=null; NoiseSuppressor ns=null; AutomaticGainControl agc=null;
        short[] pcm=new short[frameSamples]; float[] before=new float[CAPTURE_FRAMES*frameSamples], after=new float[(CAPTURE_FRAMES+32)*FRAME];
        double[] timings=new double[CAPTURE_FRAMES]; int count=0, latency=0; boolean systemEffectsDisabled=false; int session=recorder.getAudioSessionId();
        ByteBuffer mic=NativeEngine.floats(frameSamples),out=NativeEngine.floats(FRAME),stats=NativeEngine.floats(16);
        final int modules=NativeEngine.NS|NativeEngine.AGC|NativeEngine.LIMITER|(microphones>1 ? NativeEngine.BF : 0);
        try(NativeEngine engine=new NativeEngine(RATE,FRAME,microphones,modules,0,0,delays,NativeEngine.PROFILE_VEHICLE_STREAM,noiseBackend)) {
            if(recorder.getState()!=AudioRecord.STATE_INITIALIZED || recorder.getChannelCount()!=microphones || recorder.getSampleRate()!=RATE)
                throw new IllegalStateException("device did not expose selected synchronous microphone channels");
            if(AcousticEchoCanceler.isAvailable()) aec=AcousticEchoCanceler.create(session);
            if(NoiseSuppressor.isAvailable()) ns=NoiseSuppressor.create(session);
            if(AutomaticGainControl.isAvailable()) agc=AutomaticGainControl.create(session);
            systemEffectsDisabled=disable(aec)&disable(ns)&disable(agc);
            show("正在采集 10 秒……"); recorder.startRecording();
            while(count<CAPTURE_FRAMES && !stopCapture) {
                int got=0;
                while(got<frameSamples) {int n=recorder.read(pcm,got,frameSamples-got,AudioRecord.READ_BLOCKING); if(n<=0) throw new IllegalStateException("AudioRecord read failed: "+n); got+=n;}
                for(int i=0;i<frameSamples;i++) {float v=pcm[i]/32768f; before[count*frameSamples+i]=v; mic.putFloat(i*4,v);}
                long t=System.nanoTime(); int result=engine.process(mic,null,out,stats); timings[count]=(System.nanoTime()-t)/1e6;
                if(result!=0) throw new IllegalStateException("native frame failed: "+result);
                latency=stats.getInt(36);
                for(int i=0;i<FRAME;i++) after[count*FRAME+i]=out.getFloat(i*4);
                count++;
            }
            recorder.stop();
            for(int i=0;i<frameSamples;i++) mic.putFloat(i*4,0);
            int drainFrames=(latency+FRAME-1)/FRAME;
            if(drainFrames>32) throw new IllegalStateException("drain buffer capacity");
            for(int k=0;k<drainFrames;k++) {if(engine.process(mic,null,out,stats)!=0) throw new IllegalStateException("drain failed"); for(int i=0;i<FRAME;i++) after[(count+k)*FRAME+i]=out.getFloat(i*4);}
        } finally {
            if(recorder.getRecordingState()==AudioRecord.RECORDSTATE_RECORDING) recorder.stop();
            recorder.release(); if(aec!=null) aec.release(); if(ns!=null) ns.release(); if(agc!=null) agc.release();
        }
        Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT);
        if(count==0) throw new IllegalStateException("capture cancelled before first frame");
        writeWav(new File(getFilesDir(),"capture.before.wav"),before,count*frameSamples,microphones,RATE);
        writeWav(new File(getFilesDir(),"capture.after.wav"),after,count*FRAME+latency,1,RATE);
        double[] measured=Arrays.copyOf(timings,count); Arrays.sort(measured); int misses=0; for(double v:measured) if(v>10) misses++;
        JSONObject report=new JSONObject(); report.put("schema_version",2); report.put("webrtc_revision",NativeEngine.revision()); report.put("input_kind","device microphone PCM16"); report.put("sample_rate",RATE); report.put("frame_size",FRAME); report.put("frames",count);
        report.put("microphones",microphones); report.put("output_channels",1); report.put("beamforming_enabled",microphones>1);
        report.put("beam_delays_samples",new org.json.JSONArray(delays)); report.put("beam_scope","configured delay-and-sum; zero delays select broadside; array geometry not verified");
        report.put("source_requested",unprocessed?"UNPROCESSED":"VOICE_RECOGNITION"); report.put("system_effect_disable_calls_succeeded",systemEffectsDisabled);
        report.put("hardware_raw_capture_verified",false); report.put("system_effects_note","disabled controllable effects when present; vendor preprocessing is not proven absent");
        report.put("processing_profile","vehicle_stream"); report.put("ns_level",0);
        String nsName=noiseBackend==2 ? "SpeexDSP" : noiseBackend==1 ? "RNNoise" : "WebRTC NS";
        report.put("ns_backend",noiseBackend==2 ? "speex" : noiseBackend==1 ? "rnnoise" : "webrtc");
        report.put("processing_policy",RATE==16000 && noiseBackend!=1 ? "legacy_16k" : "native_fullband");
        report.put("modules",(microphones>1 ? "delay-and-sum BF," : "")+nsName+",AGC2,limiter"); report.put("dsp_delay_samples",latency); report.put("noise_estimation","online adaptive"); report.put("timing_scope","Java validation + JNI + C++ process; excludes AudioRecord read and OS latency");
        report.put("p50_ms",measured[(count-1)/2]); report.put("p95_ms",measured[(int)Math.ceil(.95*count)-1]); report.put("max_ms",measured[count-1]); report.put("deadline_misses",misses);
        report.put("aec_bf_dtd_validated_on_capture",false); report.put("standard_compliance",false); save("capture.json",report);
        show("真实麦克风处理完成："+count+" 帧\nP95 "+report.getDouble("p95_ms")+" ms\n已保存处理前/后 WAV 和 capture.json\n应用私有目录："+getFilesDir());
    }
    private static void writeWav(File file,float[] samples,int length) throws Exception {
        writeWav(file,samples,length,1);
    }
    private static void writeWav(File file,float[] samples,int length,int channels) throws Exception {
        writeWav(file,samples,length,channels,RATE);
    }
    private static void writeWav(File file,float[] samples,int length,int channels,int rate) throws Exception {
        ByteBuffer bytes=ByteBuffer.allocate(44+length*2).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36+length*2).put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16);
        bytes.putShort((short)1).putShort((short)channels).putInt(rate).putInt(rate*channels*2).putShort((short)(channels*2)).putShort((short)16);
        bytes.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(length*2);
        for(int i=0;i<length;i++) bytes.putShort((short)Math.round(Math.max(-1f,Math.min(32767f/32768f,samples[i]))*32768));
        try(FileOutputStream f=new FileOutputStream(file)) {f.write(bytes.array());}
    }
    @Override protected void onPause() {stopCapture=true; super.onPause();}
}
