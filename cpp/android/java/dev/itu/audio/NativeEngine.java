package dev.itu.audio;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** One engine per stream. Stop/join the processing thread before reset or close.
 * Buffers: direct native-order float32, position 0, full limit, distinct storage; borrowed only during process.
 * Statistics: 64 bytes, frameIndex@0(long), flags@8(int), RMS@12/16/20(float),
 * ERL/ERLE@24/28(float), limitedSamples@32(int), DSP delay@36(int),
 * statisticsValid@40(int: ERL=1, ERLE=2, delay=4), estimatedDelayMs@44(int).
 * Valid bits mean upstream field presence, not convergence; delay 0 may be unknown.
 * AEC nonlinear delay@52(int) in I/O-rate samples; linear delay@56(int) always in 16 kHz samples.
 * Flags: far=1, linearAecAvailable=2. DTD/adaptation status is unavailable.
 * I/O supports 16/24/48 kHz, exactly 160/240/480 samples/channel per 10 ms call.
 * Tap buffers: AEC interleaved after residual suppression; linear tap always 160 samples/channel at 16 kHz.
 */
public final class NativeEngine implements AutoCloseable {
    public static final int AEC=1, BF=2, NS=8, AGC=16, LIMITER=64;
    public static final int PROFILE_GENERAL=0, PROFILE_REFERENCED_RADIO=1, PROFILE_VEHICLE_STREAM=2;
    public static final int NS_WEBRTC=0, NS_RNNOISE=1, NS_SPEEX=2, NS_ULUNAS=3;
    public static final int COLD_NOISE_CONTROL=1, SPATIAL_MVDR=2, WIND_CONTROL=4;
    static { System.loadLibrary("itu_audio_native"); }
    private long handle;
    private final int micBytes, frameBytes, linearBytes, referenceBytes;
    private final boolean needsReference;
    public NativeEngine(int rate,int frame,int microphones,int modules,int noiseLevel,int streamDelayMs,float[] delays) {
        this(rate,frame,microphones,modules,noiseLevel,streamDelayMs,delays,PROFILE_GENERAL);
    }
    public NativeEngine(int rate,int frame,int microphones,int modules,int noiseLevel,int streamDelayMs,float[] delays,int profile) {
        this(rate,frame,microphones,modules,noiseLevel,streamDelayMs,delays,profile,NS_WEBRTC);
    }
    public NativeEngine(int rate,int frame,int microphones,int modules,int noiseLevel,int streamDelayMs,float[] delays,int profile,int noiseBackend) {
        this(rate,frame,microphones,modules,noiseLevel,streamDelayMs,delays,profile,noiseBackend,1,0);
    }
    public NativeEngine(int rate,int frame,int microphones,int modules,int noiseLevel,int streamDelayMs,float[] delays,int profile,int noiseBackend,int references,int extensions) {
        this(rate,frame,microphones,modules,noiseLevel,streamDelayMs,delays,profile,noiseBackend,references,extensions,null);
    }
    public NativeEngine(int rate,int frame,int microphones,int modules,int noiseLevel,int streamDelayMs,float[] delays,int profile,int noiseBackend,int references,int extensions,String modelPath) {
        handle=nativeCreateWithModel(rate,frame,microphones,modules,noiseLevel,streamDelayMs,delays,profile,noiseBackend,references,extensions,modelPath);
        if(handle==0) throw new IllegalStateException("native creation failed");
        micBytes=frame*microphones*4; frameBytes=frame*4; linearBytes=160*microphones*4; needsReference=(modules&AEC)!=0;
        referenceBytes=frame*references*4;
    }
    public static ByteBuffer floats(int samples) { return ByteBuffer.allocateDirect(samples*4).order(ByteOrder.nativeOrder()); }
    private static boolean valid(ByteBuffer b,int bytes,boolean writable) {
        return b!=null && b.isDirect() && b.capacity()>=bytes && b.capacity()%4==0 && b.position()==0 &&
            b.limit()==b.capacity() && b.order()==ByteOrder.nativeOrder() && (!writable || !b.isReadOnly());
    }
    public int process(ByteBuffer microphones,ByteBuffer reference,ByteBuffer output,ByteBuffer stats) {
        return processWithTaps(microphones,reference,output,stats,null,null,null);
    }
    public int processWithTaps(ByteBuffer microphones,ByteBuffer reference,ByteBuffer output,ByteBuffer stats,ByteBuffer aecTap,ByteBuffer bfTap,ByteBuffer linearTap) {
        if(handle==0) return -1;
        if(output==stats || microphones==stats || reference==stats || !valid(microphones,micBytes,false) || !valid(output,frameBytes,true) || !valid(stats,64,true) ||
            (reference!=null && !valid(reference,referenceBytes,false)) || (needsReference && reference==null)) return -2;
        if((aecTap!=null && !valid(aecTap,micBytes,true)) || (bfTap!=null && !valid(bfTap,frameBytes,true)) ||
            (linearTap!=null && !valid(linearTap,linearBytes,true))) return -2;
        if((aecTap!=null && (aecTap==microphones || aecTap==reference || aecTap==output || aecTap==stats || aecTap==bfTap || aecTap==linearTap)) ||
            (bfTap!=null && (bfTap==microphones || bfTap==reference || bfTap==output || bfTap==stats || bfTap==linearTap)) ||
            (linearTap!=null && (linearTap==microphones || linearTap==reference || linearTap==output || linearTap==stats))) return -2;
        return nativeProcess(handle,microphones,reference,output,stats,aecTap,bfTap,linearTap);
    }
    public long workingMemoryBytes() {return nativeMemory(handle);}
    public void reset() {if(handle==0) throw new IllegalStateException("engine closed"); nativeReset(handle);}
    public void close() {long h=handle; handle=0; nativeDestroy(h);}
    public static String revision() {return nativeRevision();}
    public static boolean backendAvailable(int backend) {return nativeBackendAvailable(backend);}
    private static native boolean nativeBackendAvailable(int backend);
    private static native long nativeCreate(int rate,int frame,int channels,int flags,int noiseLevel,int streamDelayMs,float[] delays);
    private static native long nativeCreateWithProfile(int rate,int frame,int channels,int flags,int noiseLevel,int streamDelayMs,float[] delays,int profile);
    private static native long nativeCreateWithBackend(int rate,int frame,int channels,int flags,int noiseLevel,int streamDelayMs,float[] delays,int profile,int noiseBackend);
    private static native long nativeCreateExtended(int rate,int frame,int channels,int flags,int noiseLevel,int streamDelayMs,float[] delays,int profile,int noiseBackend,int references,int extensions);
    private static native long nativeCreateWithModel(int rate,int frame,int channels,int flags,int noiseLevel,int streamDelayMs,float[] delays,int profile,int noiseBackend,int references,int extensions,String modelPath);
    private static native int nativeProcess(long h,ByteBuffer mic,ByteBuffer ref,ByteBuffer output,ByteBuffer stats,ByteBuffer aecTap,ByteBuffer bfTap,ByteBuffer linearTap);
    private static native void nativeDestroy(long h);
    private static native void nativeReset(long h);
    private static native long nativeMemory(long h);
    private static native String nativeRevision();
}
