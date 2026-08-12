using System.Runtime.InteropServices;

namespace FirefoxLoopbackCapture;

[StructLayout(LayoutKind.Sequential)]
struct WAVEFORMATEX
{
    public ushort wFormatTag;
    public ushort nChannels;
    public uint nSamplesPerSec;
    public uint nAvgBytesPerSec;
    public ushort nBlockAlign;
    public ushort wBitsPerSample;
    public ushort cbSize;
}

[StructLayout(LayoutKind.Sequential)]
struct AUDIOCLIENT_ACTIVATION_PARAMS
{
    public int ActivationType;      // AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
    public uint ProcessId;
    public int ProcessLoopbackMode; // INCLUDE_TARGET_PROCESS_TREE = 0, EXCLUDE = 1
}

// PROPVARIANT wrapping a VT_BLOB pointing at AUDIOCLIENT_ACTIVATION_PARAMS.
// All fields are blittable (ushort/uint/IntPtr) so the default P/Invoke marshaler
// can pass this by value without needing a manual unmanaged copy.
[StructLayout(LayoutKind.Explicit, Size = 24)]
struct PROPVARIANT
{
    [FieldOffset(0)] public ushort vt;
    [FieldOffset(8)] public uint blobSize;
    [FieldOffset(16)] public IntPtr blobData;
}

[ComImport, Guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioClient
{
    void Initialize(int shareMode, int streamFlags, long hnsBufferDuration, long hnsPeriodicity, ref WAVEFORMATEX format, IntPtr audioSessionGuid);
    void GetBufferSize(out uint numBufferFrames);
    void GetStreamLatency(out long latency);
    void GetCurrentPadding(out uint numPaddingFrames);
    void IsFormatSupported(int shareMode, ref WAVEFORMATEX format, IntPtr closestMatch);
    void GetMixFormat(out IntPtr deviceFormat);
    void GetDevicePeriod(out long defaultDevicePeriod, out long minimumDevicePeriod);
    void Start();
    void Stop();
    void Reset();
    void SetEventHandle(IntPtr eventHandle);
    void GetService(ref Guid riid, [MarshalAs(UnmanagedType.IUnknown)] out object service);
}

[ComImport, Guid("C8ADBD64-E71E-48A0-A4DE-185C395CD317"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioCaptureClient
{
    void GetBuffer(out IntPtr data, out uint numFramesToRead, out uint flags, out ulong devicePosition, out ulong qpcPosition);
    void ReleaseBuffer(uint numFramesRead);
    void GetNextPacketSize(out uint numFramesInNextPacket);
}

[ComImport, Guid("72A22D78-CDE4-431D-B8CC-843A71199B6D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IActivateAudioInterfaceAsyncOperation
{
    void GetActivateResult(out int activateResult, [MarshalAs(UnmanagedType.IUnknown)] out object activatedInterface);
}

[ComImport, Guid("41D949AB-9862-444A-80F6-C261334DA5EB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IActivateAudioInterfaceCompletionHandler
{
    void ActivateCompleted(IActivateAudioInterfaceAsyncOperation activateOperation);
}

static class NativeMethods
{
    public const string VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback";

    public const int AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1;
    public const int PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0;

    public const int AUDCLNT_SHAREMODE_SHARED = 0;
    public const int AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000;
    public const uint AUDCLNT_BUFFERFLAGS_SILENT = 0x2;
    public const ushort WAVE_FORMAT_IEEE_FLOAT = 3;
    public const ushort VT_BLOB = 0x41;

    [DllImport("Mmdevapi.dll", PreserveSig = true)]
    public static extern int ActivateAudioInterfaceAsync(
        [MarshalAs(UnmanagedType.LPWStr)] string deviceInterfacePath,
        [In] ref Guid riid,
        [In] ref PROPVARIANT activationParams,
        IActivateAudioInterfaceCompletionHandler completionHandler,
        out IActivateAudioInterfaceAsyncOperation activationOperation);
}

sealed class CompletionHandler : IActivateAudioInterfaceCompletionHandler
{
    public readonly ManualResetEventSlim Done = new(false);
    public object? ActivatedInterface;
    public int Hr;

    public void ActivateCompleted(IActivateAudioInterfaceAsyncOperation activateOperation)
    {
        activateOperation.GetActivateResult(out Hr, out var iface);
        ActivatedInterface = iface;
        Done.Set();
    }
}
