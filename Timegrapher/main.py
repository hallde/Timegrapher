from operator import indexOf
import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter, find_peaks, filtfilt

#Normal settings
duration = 20
hz = 0


fs = 128000
lowcut = 1000
highcut = 3000
epsilon = 1e-6
peak_db_threshold = -20



def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    y = filtfilt(b, a, data)
    return y

def analyze_beat_error(intervals):
    sum_error = 0.0
    error_calcs = 0
    for i in range(1,len(intervals),2):
        error = np.abs(intervals[i-1]-intervals[i])
        sum_error += error
        error_calcs += 1
    if error_calcs > 0:
        return sum_error/error_calcs


def beat_rate_auto_detect():
    print("Detecting beat rate...")
    common_beats=[36000,28800,21600,19800,18000,14400,10800,7200,3600]
    audio = sd.rec(int(10 * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    filtered = bandpass_filter(audio, lowcut, highcut, fs)
    envelope = np.abs(filtered)
    envelope = envelope / (np.max(envelope) + epsilon)
    db_envelope = 20 * np.log10(envelope + epsilon)
    for beat_rate in common_beats:
        min_tick_spacing=(3600/beat_rate)/2

        peaks, _ = find_peaks(db_envelope, height=peak_db_threshold, distance=min_tick_spacing * fs)
        tick_times = peaks / fs
        intervals = np.diff(tick_times)
        std_interval = np.std(intervals)
        mean_interval = np.mean(intervals)
        bps_target = 1/(beat_rate / 3600)
        print(f"Actual: {mean_interval} | Comparison: {bps_target} | High end: {mean_interval+std_interval} | Low End: {mean_interval-std_interval}")

        if mean_interval-(std_interval*0.5) <= bps_target <= (std_interval*0.5)+mean_interval:
            print("Flagged")
            return beat_rate
    print("Flag missed - defaulting to 3600")
    return 3600

if hz == 0:
    hz=(beat_rate_auto_detect())
    print(f"Beat rate set to {hz}bph")
    min_tick_spacing = (3600/hz)/2

analyze_beat_error([1,2,2,3,3,1,43,3,3,3,3,3,3,3,3,1])
print("Recording...")
audio = sd.rec(int(duration*fs), samplerate=fs, channels=1)
sd.wait()
audio = audio.flatten()

filtered = bandpass_filter(audio, lowcut, highcut, fs)
envelope = np.abs(filtered)
envelope = envelope / (np.max(envelope)+epsilon)
db_envelope = 20 * np.log10(envelope + epsilon)
peaks, _ = find_peaks(db_envelope, height=peak_db_threshold, distance=min_tick_spacing*fs)
tick_times = peaks / fs



if len(tick_times) >= 2:
    intervals = np.diff(tick_times)
    Q1 = np.percentile(intervals, 25, method='midpoint')
    Q3 = np.percentile(intervals, 75, method='midpoint')
    #Do we really wanna do IQR based stuff, or just standard deviation from expected time between beats???
    IQR = Q3 - Q1
    print(IQR)
    upper = Q3 + IQR
    lower = Q1 - IQR

    mask = (intervals >= lower) & (intervals <= upper)
    intervals_clean = intervals[mask]

    avg_interval = np.mean(intervals_clean)
    min_interval = np.min(intervals_clean)
    max_interval = np.max(intervals_clean)
    std_interval = np.std(intervals_clean)

    gain_per_day = ((3600/hz) - avg_interval) * 86400


    print(f"tick intervals without scrubbing: {intervals}")
    print(f"tick intervals with scrubbing: {intervals_clean}")
    print(f"Average tick interval: {avg_interval:.4f}s")

    print(f"Detected {len(intervals)} ticks in {duration}s, {len(intervals_clean)} intervals were valid")

    print(f"Min interval: {min_interval:.4f}s, Max interval: {max_interval:.4f}s, Std: {std_interval:.4f}s")
    print(f"Estimated daily gain/loss w/ scrub: {gain_per_day:.2f}s/day")
    print(f"Estimated daily gain/loss w/o scrub: {((3600/hz) - np.mean(intervals)) * 86400:.2f}s/day")
    print(f"Beat error w/ scrub: {analyze_beat_error(intervals_clean)*1000:.4f}ms")
    print(f"Beat error w/o scrub: {analyze_beat_error(intervals)*1000:.4f}ms")

else:
    print("Not enough ticks detected for analysis")

