using System.Diagnostics;
using System.Runtime.InteropServices;
using FirefoxLoopbackCapture;

string processName = args.Length > 0 ? args[0] : "firefox";

var candidates = Process.GetProcessesByName(processName);
if (candidates.Length == 0)
{
    Console.Error.WriteLine($"ERROR: no running process named '{processName}.exe' found");
    return 1;
}

// Firefox's content/GPU/RDD child processes also run as firefox.exe on Windows, so filter
// down to the process that owns a visible top-level window (the main browser process).
// Process-loopback capture on that PID pulls in the whole process tree, including the
// child process that's actually rendering audio.
var target = candidates.FirstOrDefault(p => p.MainWindowHandle != IntPtr.Zero) ?? candidates[0];
uint targetPid = (uint)target.Id;
Console.Error.WriteLine($"INFO: targeting {processName}.exe pid={targetPid}");

var activationParams = new AUDIOCLIENT_ACTIVATION_PARAMS
{
    ActivationType = NativeMethods.AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
    ProcessId = targetPid,
    ProcessLoopbackMode = NativeMethods.PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
};

IntPtr paramsPtr = Marshal.AllocHGlobal(Marshal.SizeOf<AUDIOCLIENT_ACTIVATION_PARAMS>());
Marshal.StructureToPtr(activationParams, paramsPtr, false);

var pv = new PROPVARIANT
{
    vt = NativeMethods.VT_BLOB,
    blobSize = (uint)Marshal.SizeOf<AUDIOCLIENT_ACTIVATION_PARAMS>(),
    blobData = paramsPtr,
};

var handler = new CompletionHandler();
Guid iidAudioClient = typeof(IAudioClient).GUID;

int hr = NativeMethods.ActivateAudioInterfaceAsync(
    NativeMethods.VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
    ref iidAudioClient,
    ref pv,
    handler,
    out _);
Marshal.ThrowExceptionForHR(hr);

handler.Done.Wait();
Marshal.ThrowExceptionForHR(handler.Hr);
var audioClient = (IAudioClient)handler.ActivatedInterface!;
Marshal.FreeHGlobal(paramsPtr);

const int SAMPLE_RATE = 48000;
const int CHANNELS = 2;
const int BITS_PER_SAMPLE = 32;
int blockAlign = CHANNELS * (BITS_PER_SAMPLE / 8);

var format = new WAVEFORMATEX
{
    wFormatTag = NativeMethods.WAVE_FORMAT_IEEE_FLOAT,
    nChannels = CHANNELS,
    nSamplesPerSec = SAMPLE_RATE,
    nAvgBytesPerSec = (uint)(SAMPLE_RATE * blockAlign),
    nBlockAlign = (ushort)blockAlign,
    wBitsPerSample = BITS_PER_SAMPLE,
    cbSize = 0,
};

const long HNS_BUFFER_DURATION = 2_000_000; // 200ms in 100ns units
audioClient.Initialize(
    NativeMethods.AUDCLNT_SHAREMODE_SHARED,
    NativeMethods.AUDCLNT_STREAMFLAGS_LOOPBACK,
    HNS_BUFFER_DURATION,
    0,
    ref format,
    IntPtr.Zero);

Guid iidCaptureClient = typeof(IAudioCaptureClient).GUID;
audioClient.GetService(ref iidCaptureClient, out var captureObj);
var captureClient = (IAudioCaptureClient)captureObj;

// Header line for the Python side to know how to interpret/resample the raw stream.
Console.Error.WriteLine($"FORMAT rate={SAMPLE_RATE} channels={CHANNELS} sample_fmt=f32le");
Console.Error.Flush();

using var stdout = Console.OpenStandardOutput();

audioClient.Start();

var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cts.Cancel();
};

try
{
    while (!cts.IsCancellationRequested)
    {
        captureClient.GetNextPacketSize(out uint packetLength);
        while (packetLength != 0)
        {
            captureClient.GetBuffer(out IntPtr data, out uint framesAvailable, out uint flags, out _, out _);
            int byteCount = (int)(framesAvailable * blockAlign);
            if (byteCount > 0)
            {
                byte[] buffer = new byte[byteCount];
                if ((flags & NativeMethods.AUDCLNT_BUFFERFLAGS_SILENT) == 0)
                {
                    Marshal.Copy(data, buffer, 0, byteCount);
                }
                stdout.Write(buffer, 0, byteCount);
                stdout.Flush();
            }
            captureClient.ReleaseBuffer(framesAvailable);
            captureClient.GetNextPacketSize(out packetLength);
        }
        Thread.Sleep(10);
    }
}
finally
{
    audioClient.Stop();
}

return 0;
