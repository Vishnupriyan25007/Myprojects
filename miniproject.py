# ================================================================
# FPGA SPEECH DENOISING PROJECT
# DTCWT + BIVARIATE SHRINKAGE + SPEECH/NOISE SEPARATION
#
# Input:
#   WhatsApp Audio 2026-10-06 at 6.48.55 PM (mp3cut.net) (1).mp3
#
# Output:
#   final_denoised_speech.wav
#
# Main modules retained:
#   - imageio_ffmpeg
#   - soundfile
#   - numpy
#   - dtcwt
#   - scipy
#
# ================================================================

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import imageio_ffmpeg
import dtcwt

from scipy import signal


# ================================================================
# USER SETTINGS
# ================================================================

INPUT_FILE = (
    r"C:\Users\Vishnupriyan\Desktop"
    r"\WhatsApp Audio 2026-10-06 at 6.48.55 PM.mpeg"
)

OUTPUT_FILE = (
    r"C:\Users\Vishnupriyan\Desktop"
    r"\final_denoised_speech.wav"
)

FS = 16000
CHANNELS = 1
FRAME_SIZE = 512

# DTCWT
DTCWT_LEVELS = 3

# Noise estimation
MAD_CONSTANT = 0.6745

# Bivariate shrinkage strength
# Higher = stronger suppression
SHRINK_STRENGTH = 1.25

# Extra suppression in noise-only regions
NOISE_ONLY_SUPPRESSION = 0.08

# Minimum gain allowed in speech regions
MIN_SPEECH_GAIN = 0.72

# Maximum restoration gain
MAX_RESTORE_GAIN = 1.35

# STFT parameters
STFT_NFFT = 1024
STFT_HOP = 256

# ================================================================
# IMPORTANT:
# Do NOT use a fixed final multiplication such as:
#
# output = output * 0.5
#
# because that only reduces volume.
#
# Instead, speech-level restoration is calculated from the
# original and denoised signals.
# ================================================================


# ================================================================
# STEP 1
# CONVERT INPUT AUDIO TO 16 kHz MONO 16-bit WAV
# ================================================================

def convert_input_to_wav(input_file, output_wav):

    print("\n================================================")
    print("STEP 1 - INPUT AUDIO CONVERSION")
    print("================================================")

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            "\nInput audio file not found:\n" + input_file
        )

    print("Input:")
    print(input_file)
    print("Status: FOUND")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_file,
        "-ac",
        str(CHANNELS),
        "-ar",
        str(FS),
        "-c:a",
        "pcm_s16le",
        output_wav
    ]

    try:

        # IMPORTANT:
        # Do not use text=True.
        # FFmpeg may produce non-UTF/CP1252 bytes.
        # This avoids the UnicodeDecodeError encountered earlier.
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

    except subprocess.CalledProcessError as e:

        print("\nFFmpeg conversion failed.")

        try:
            error_text = e.stderr.decode(
                "utf-8",
                errors="replace"
            )
        except Exception:
            error_text = str(e)

        print(error_text)

        raise RuntimeError(
            "Could not convert input audio."
        )

    if not os.path.exists(output_wav):
        raise FileNotFoundError(
            "Converted WAV file was not created."
        )

    audio, fs = sf.read(
        output_wav,
        dtype="int16"
    )

    print("\nConversion successful.")
    print("Sampling frequency :", fs)
    print("Samples            :", len(audio))
    print("Channels           :", CHANNELS)
    print("Format             : 16-bit PCM")

    return output_wav


# ================================================================
# STEP 2
# LOAD AUDIO
# ================================================================

def load_audio(filename):

    audio, fs = sf.read(
        filename,
        dtype="int16"
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio.astype(np.float64),
            axis=1
        )
    else:
        audio = audio.astype(np.float64)

    if fs != FS:
        raise ValueError(
            f"Expected {FS} Hz but found {fs} Hz."
        )

    return audio


# ================================================================
# STEP 3
# FRAME EXTRACTION
# ================================================================

def create_frames(audio, frame_size):

    frames = []

    number_of_frames = int(
        np.ceil(len(audio) / frame_size)
    )

    for i in range(number_of_frames):

        start = i * frame_size
        end = min(
            start + frame_size,
            len(audio)
        )

        frame = audio[start:end]

        if len(frame) < frame_size:

            padded = np.zeros(
                frame_size,
                dtype=np.float64
            )

            padded[:len(frame)] = frame
            frame = padded

        frames.append(frame)

    return frames


# ================================================================
# STEP 4
# FRAME RMS
# ================================================================

def calculate_rms(frame):

    return np.sqrt(
        np.mean(frame ** 2) + 1e-12
    )


# ================================================================
# STEP 5
# SPEECH ACTIVITY ESTIMATION
#
# This is intentionally conservative.
#
# We estimate:
#
#   speech_present = True
#       -> preserve speech
#
#   speech_present = False
#       -> aggressively suppress non-speech noise
#
# ================================================================

def estimate_speech_activity(frame, reference_rms):

    rms = calculate_rms(frame)

    if reference_rms <= 1e-12:
        return False

    ratio = rms / reference_rms

    # Conservative threshold
    if ratio > 1.45:
        return True

    return False


# ================================================================
# STEP 6
# ESTIMATE GLOBAL NOISE LEVEL
#
# Use the lower-energy frames to estimate background/noise level.
# ================================================================

def estimate_frame_noise_level(audio):

    frames = create_frames(
        audio,
        FRAME_SIZE
    )

    rms_values = np.array([
        calculate_rms(frame)
        for frame in frames
    ])

    if len(rms_values) == 0:
        return 1.0

    # Lowest 20% frames
    percentile = np.percentile(
        rms_values,
        20
    )

    low_energy = rms_values[
        rms_values <= percentile
    ]

    if len(low_energy) == 0:
        low_energy = rms_values

    noise_rms = np.median(
        low_energy
    )

    return max(
        noise_rms,
        1.0
    )


# ================================================================
# STEP 7
# MAD NOISE ESTIMATION
#
# Finest DTCWT detail coefficients are used.
# ================================================================

def estimate_mad_sigma(level1):

    magnitude = np.abs(level1)

    median_value = np.median(
        magnitude
    )

    mad = np.median(
        np.abs(
            magnitude - median_value
        )
    )

    sigma = mad / MAD_CONSTANT

    return max(
        sigma,
        1e-8
    )


# ================================================================
# STEP 8
# BIVARIATE SHRINKAGE
#
# Child:
#   finer-scale coefficient
#
# Parent:
#   corresponding coarser-scale coefficient
#
# Joint magnitude:
#
#       R = sqrt(|child|^2 + |parent|^2)
#
# Gain:
#
#       G = max(0, 1 - T/R)
#
# Output:
#
#       y = G * child
#
# This preserves coefficients that are supported
# across scales and suppresses isolated coefficients.
# ================================================================

def bivariate_shrinkage(
    child,
    parent,
    threshold,
    strength=1.0
):

    R = np.sqrt(
        np.abs(child) ** 2 +
        np.abs(parent) ** 2
    )

    effective_threshold = (
        threshold * strength
    )

    if R <= effective_threshold:

        gain = 0.0

    else:

        gain = (
            R - effective_threshold
        ) / (R + 1e-12)

    output = gain * child

    return output


# ================================================================
# STEP 9
# DTCWT + BIVARIATE DENOISING
# ================================================================

def dtcwt_bivariate_denoise(
    frame,
    strength=SHRINK_STRENGTH
):

    frame = frame.astype(
        np.float64
    )

    transform = dtcwt.Transform1d()

    result = transform.forward(
        frame,
        nlevels=DTCWT_LEVELS
    )

    level1 = result.highpasses[0][:, 0]
    level2 = result.highpasses[1][:, 0]
    level3 = result.highpasses[2][:, 0]

    # ------------------------------------------------------------
    # Noise estimate from finest level
    # ------------------------------------------------------------

    sigma = estimate_mad_sigma(
        level1
    )

    # Adaptive threshold
    T1 = sigma * 1.00
    T2 = sigma * 0.85

    # ------------------------------------------------------------
    # Level 1 -> Level 2
    # ------------------------------------------------------------

    output_level1 = np.zeros_like(
        level1
    )

    for i in range(len(level1)):

        parent_index = i // 2

        if parent_index >= len(level2):
            parent_index = len(level2) - 1

        parent = level2[
            parent_index
        ]

        output_level1[i] = (
            bivariate_shrinkage(
                level1[i],
                parent,
                T1,
                strength
            )
        )

    # ------------------------------------------------------------
    # Level 2 -> Level 3
    # ------------------------------------------------------------

    output_level2 = np.zeros_like(
        level2
    )

    for i in range(len(level2)):

        parent_index = i // 2

        if parent_index >= len(level3):
            parent_index = len(level3) - 1

        parent = level3[
            parent_index
        ]

        output_level2[i] = (
            bivariate_shrinkage(
                level2[i],
                parent,
                T2,
                strength
            )
        )

    # ------------------------------------------------------------
    # Level 3
    #
    # No coarser high-pass parent is available.
    #
    # Do NOT destroy Level 3 because it contains important
    # low-frequency speech structure.
    # ------------------------------------------------------------

    output_level3 = level3.copy()

    # ------------------------------------------------------------
    # Reconstruct DTCWT pyramid
    # ------------------------------------------------------------

    highpasses = [
        output_level1.reshape(-1, 1),
        output_level2.reshape(-1, 1),
        output_level3.reshape(-1, 1)
    ]

    pyramid = dtcwt.Pyramid(
        lowpass=result.lowpass,
        highpasses=highpasses
    )

    denoised = transform.inverse(
        pyramid
    )

    denoised = np.real(
        denoised
    )

    return denoised


# ================================================================
# STEP 10
# NOISE-ONLY TIME-DOMAIN SUPPRESSION
#
# Important:
#
# This is applied only when the frame is classified as
# noise-dominant.
#
# Speech-active frames are not multiplied by the same factor.
# ================================================================

def suppress_noise_only_frames(
    original,
    denoised,
    noise_rms
):

    output = denoised.copy()

    frames_original = create_frames(
        original,
        FRAME_SIZE
    )

    frames_denoised = create_frames(
        denoised,
        FRAME_SIZE
    )

    result = np.zeros_like(
        original,
        dtype=np.float64
    )

    number_of_frames = len(
        frames_original
    )

    # Speech/noise reference threshold
    speech_threshold = (
        noise_rms * 1.35
    )

    for i in range(number_of_frames):

        original_frame = (
            frames_original[i]
        )

        denoised_frame = (
            frames_denoised[i]
        )

        rms = calculate_rms(
            original_frame
        )

        # --------------------------------------------------------
        # Noise-only frame
        # --------------------------------------------------------

        if rms <= speech_threshold:

            processed = (
                denoised_frame *
                NOISE_ONLY_SUPPRESSION
            )

        # --------------------------------------------------------
        # Speech-active frame
        # --------------------------------------------------------

        else:

            processed = (
                denoised_frame
            )

        start = (
            i * FRAME_SIZE
        )

        end = min(
            start + FRAME_SIZE,
            len(original)
        )

        result[start:end] = (
            processed[:end-start]
        )

    return result


# ================================================================
# STEP 11
# SPECTRAL NOISE PROFILE
#
# This is used to suppress persistent noise components.
#
# IMPORTANT:
# The noise profile is reshaped to:
#
#       (frequency_bins, 1)
#
# so that it broadcasts correctly against:
#
#       (frequency_bins, frames)
#
# This fixes your previous:
#
# ValueError:
# operands could not be broadcast together with
# shapes (513,1476) (513,)
# ================================================================

def create_noise_profile(
    audio,
    noise_rms
):

    frequencies, times, Zxx = signal.stft(
        audio,
        fs=FS,
        nperseg=STFT_NFFT,
        noverlap=STFT_NFFT - STFT_HOP,
        boundary="zeros"
    )

    magnitude = np.abs(
        Zxx
    )

    if magnitude.shape[1] == 0:
        return np.zeros(
            magnitude.shape[0]
        )

    frame_energy = np.sqrt(
        np.mean(
            magnitude ** 2,
            axis=0
        )
    )

    # Select low-energy spectral frames
    threshold = (
        noise_rms * 1.50
    )

    noise_frames = (
        frame_energy <= threshold
    )

    if np.sum(noise_frames) < 2:

        # Fall back to lowest 10%
        count = max(
            1,
            int(
                0.10 *
                magnitude.shape[1]
            )
        )

        indices = np.argsort(
            frame_energy
        )[:count]

        noise_profile = np.median(
            magnitude[:, indices],
            axis=1
        )

    else:

        noise_profile = np.median(
            magnitude[:, noise_frames],
            axis=1
        )

    return noise_profile


# ================================================================
# STEP 12
# SPECTRAL NOISE SUPPRESSION
#
# This operates only on spectral magnitude.
#
# Phase is preserved.
#
# We do NOT simply lower the complete waveform volume.
# ================================================================

def spectral_noise_suppression(
    audio,
    noise_profile,
    strength=1.0
):

    frequencies, times, Zxx = signal.stft(
        audio,
        fs=FS,
        nperseg=STFT_NFFT,
        noverlap=STFT_NFFT - STFT_HOP,
        boundary="zeros"
    )

    magnitude = np.abs(
        Zxx
    )

    phase = np.angle(
        Zxx
    )

    if magnitude.size == 0:
        return audio.copy()

    # ------------------------------------------------------------
    # FIX FOR BROADCASTING ERROR
    #
    # noise_profile:
    #       (513,)
    #
    # converted to:
    #       (513,1)
    #
    # so it can operate against:
    #       (513,1476)
    # ------------------------------------------------------------

    noise_profile = np.asarray(
        noise_profile,
        dtype=np.float64
    ).reshape(-1, 1)

    # Safety check
    if noise_profile.shape[0] != magnitude.shape[0]:

        # Interpolate profile if dimensions differ
        old_x = np.linspace(
            0,
            1,
            noise_profile.shape[0]
        )

        new_x = np.linspace(
            0,
            1,
            magnitude.shape[0]
        )

        noise_profile = np.interp(
            new_x,
            old_x,
            noise_profile[:, 0]
        ).reshape(-1, 1)

    # ------------------------------------------------------------
    # Estimate local SNR
    # ------------------------------------------------------------

    noise_power = (
        noise_profile ** 2
    )

    signal_power = (
        magnitude ** 2
    )

    # ------------------------------------------------------------
    # Wiener-like gain
    #
    # G = S/(S+N)
    #
    # This is frequency dependent.
    # ------------------------------------------------------------

    gain = (
        signal_power /
        (
            signal_power +
            strength * noise_power +
            1e-12
        )
    )

    # ------------------------------------------------------------
    # Do not completely destroy spectral components.
    #
    # This protects speech harmonics.
    # ------------------------------------------------------------

    gain = np.clip(
        gain,
        0.12,
        1.0
    )

    enhanced_magnitude = (
        magnitude * gain
    )

    enhanced_complex = (
        enhanced_magnitude *
        np.exp(1j * phase)
    )

    _, enhanced = signal.istft(
        enhanced_complex,
        fs=FS,
        nperseg=STFT_NFFT,
        noverlap=STFT_NFFT - STFT_HOP,
        input_onesided=True,
        boundary=True
    )

    enhanced = np.real(
        enhanced
    )

    # Match length
    if len(enhanced) < len(audio):

        padded = np.zeros(
            len(audio)
        )

        padded[:len(enhanced)] = (
            enhanced
        )

        enhanced = padded

    else:

        enhanced = enhanced[
            :len(audio)
        ]

    return enhanced


# ================================================================
# STEP 13
# SPEECH-PRESERVING BLENDING
#
# We do not want the spectral stage to reduce the entire
# speech signal.
#
# Therefore:
#
#   speech-active region:
#       mostly DTCWT result
#
#   noise-dominant region:
#       stronger spectral suppression
# ================================================================

def speech_preserving_blend(
    original,
    dtcwt_output,
    spectral_output,
    noise_rms
):

    result = np.zeros_like(
        original,
        dtype=np.float64
    )

    frames_original = create_frames(
        original,
        FRAME_SIZE
    )

    frames_dtcwt = create_frames(
        dtcwt_output,
        FRAME_SIZE
    )

    frames_spectral = create_frames(
        spectral_output,
        FRAME_SIZE
    )

    speech_threshold = (
        noise_rms * 1.35
    )

    for i in range(
        len(frames_original)
    ):

        original_frame = (
            frames_original[i]
        )

        d_frame = (
            frames_dtcwt[i]
        )

        s_frame = (
            frames_spectral[i]
        )

        rms = calculate_rms(
            original_frame
        )

        # --------------------------------------------------------
        # Strong speech region
        # --------------------------------------------------------

        if rms > speech_threshold * 1.5:

            processed = (
                0.82 * d_frame +
                0.18 * s_frame
            )

        # --------------------------------------------------------
        # Mixed speech/noise region
        # --------------------------------------------------------

        elif rms > speech_threshold:

            processed = (
                0.65 * d_frame +
                0.35 * s_frame
            )

        # --------------------------------------------------------
        # Noise-dominant region
        # --------------------------------------------------------

        else:

            processed = (
                0.25 * d_frame +
                0.75 * s_frame
            )

        start = (
            i * FRAME_SIZE
        )

        end = min(
            start + FRAME_SIZE,
            len(original)
        )

        result[start:end] = (
            processed[:end-start]
        )

    return result


# ================================================================
# STEP 14
# REMOVE DC OFFSET
# ================================================================

def remove_dc_offset(audio):

    return (
        audio -
        np.mean(audio)
    )


# ================================================================
# STEP 15
# SPEECH LEVEL RESTORATION
#
# IMPORTANT:
#
# This does NOT amplify the noise blindly.
#
# We calculate the RMS of the original and denoised speech-like
# regions and restore the output level conservatively.
# ================================================================

def restore_speech_level(
    original,
    denoised,
    noise_rms
):

    original = np.asarray(
        original,
        dtype=np.float64
    )

    denoised = np.asarray(
        denoised,
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # Select regions that are likely to contain speech
    # ------------------------------------------------------------

    threshold = (
        noise_rms * 1.35
    )

    mask = (
        np.abs(original) >
        threshold
    )

    # Expand mask slightly to avoid selecting only peaks
    kernel = np.ones(9)

    mask = signal.convolve(
        mask.astype(np.float64),
        kernel,
        mode="same"
    ) > 0

    if np.sum(mask) < 100:

        # Fallback
        original_rms = calculate_rms(
            original
        )

        denoised_rms = calculate_rms(
            denoised
        )

    else:

        original_rms = np.sqrt(
            np.mean(
                original[mask] ** 2
            ) + 1e-12
        )

        denoised_rms = np.sqrt(
            np.mean(
                denoised[mask] ** 2
            ) + 1e-12
        )

    if denoised_rms <= 1e-12:

        return denoised

    gain = (
        original_rms /
        denoised_rms
    )

    # Do not over-amplify
    gain = np.clip(
        gain,
        MIN_SPEECH_GAIN,
        MAX_RESTORE_GAIN
    )

    print("\nSpeech-level restoration")
    print("-------------------------")
    print(
        "Original speech RMS :",
        round(original_rms, 4)
    )
    print(
        "Denoised speech RMS :",
        round(denoised_rms, 4)
    )
    print(
        "Restoration gain    :",
        round(gain, 4)
    )

    return (
        denoised * gain
    )


# ================================================================
# STEP 16
# FINAL PEAK PROTECTION
#
# Do not normalize blindly to maximum possible amplitude.
#
# That would also increase residual noise.
# ================================================================

def protect_peak(
    audio,
    target_peak=30000.0
):

    peak = np.max(
        np.abs(audio)
    )

    if peak <= 1e-12:
        return audio

    if peak > target_peak:

        scale = (
            target_peak /
            peak
        )

        audio = (
            audio * scale
        )

    return audio


# ================================================================
# STEP 17
# SAVE FINAL WAV
# ================================================================

def save_output(
    audio,
    output_file
):

    audio = np.clip(
        audio,
        -32768,
        32767
    )

    sf.write(
        output_file,
        audio.astype(np.int16),
        FS,
        subtype="PCM_16"
    )

    print("\nOutput saved:")
    print(output_file)


# ================================================================
# STEP 18
# COMPARE SIGNAL STATISTICS
# ================================================================

def compare_signals(
    original,
    denoised
):

    N = min(
        len(original),
        len(denoised)
    )

    original = original[:N]
    denoised = denoised[:N]

    original_rms = calculate_rms(
        original
    )

    denoised_rms = calculate_rms(
        denoised
    )

    difference = (
        original -
        denoised
    )

    difference_rms = calculate_rms(
        difference
    )

    print("\n================================================")
    print("FINAL SIGNAL COMPARISON")
    print("================================================")

    print(
        "Original RMS  :",
        round(original_rms, 4)
    )

    print(
        "Denoised RMS  :",
        round(denoised_rms, 4)
    )

    print(
        "Difference RMS:",
        round(difference_rms, 4)
    )

    print(
        "Original peak :",
        round(
            np.max(
                np.abs(original)
            ),
            4
        )
    )

    print(
        "Output peak   :",
        round(
            np.max(
                np.abs(denoised)
            ),
            4
        )
    )


# ================================================================
# STEP 19
# PROCESS COMPLETE AUDIO
# ================================================================

def process_complete_audio(
    input_wav
):

    print("\n================================================")
    print("STEP 2 - LOAD INPUT AUDIO")
    print("================================================")

    audio = load_audio(
        input_wav
    )

    print(
        "Sampling frequency :",
        FS,
        "Hz"
    )

    print(
        "Number of samples  :",
        len(audio)
    )

    print(
        "Duration           :",
        round(
            len(audio) / FS,
            3
        ),
        "seconds"
    )

    # ------------------------------------------------------------
    # Estimate noise RMS
    # ------------------------------------------------------------

    noise_rms = estimate_frame_noise_level(
        audio
    )

    print("\nEstimated background RMS:")
    print(noise_rms)

    # ------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------

    frames = create_frames(
        audio,
        FRAME_SIZE
    )

    processed_frames = []

    print("\n================================================")
    print("STEP 3 - DTCWT + BIVARIATE SHRINKAGE")
    print("================================================")

    print(
        "Total frames:",
        len(frames)
    )

    for frame_number, frame in enumerate(
        frames
    ):

        processed = (
            dtcwt_bivariate_denoise(
                frame,
                strength=SHRINK_STRENGTH
            )
        )

        processed_frames.append(
            processed
        )

        if (
            frame_number % 100 == 0
        ):

            print(
                "Processed frame:",
                frame_number,
                "/",
                len(frames)
            )

    # ------------------------------------------------------------
    # Reconstruct full DTCWT output
    # ------------------------------------------------------------

    dtcwt_output = np.concatenate(
        processed_frames
    )

    dtcwt_output = dtcwt_output[
        :len(audio)
    ]

    print("\nDTCWT processing completed.")

    # ------------------------------------------------------------
    # Noise-only suppression
    # ------------------------------------------------------------

    print("\n================================================")
    print("STEP 4 - NOISE-ONLY SUPPRESSION")
    print("================================================")

    noise_only_output = (
        suppress_noise_only_frames(
            audio,
            dtcwt_output,
            noise_rms
        )
    )

    # ------------------------------------------------------------
    # Create spectral noise profile
    # ------------------------------------------------------------

    print("\n================================================")
    print("STEP 5 - SPECTRAL NOISE PROFILE")
    print("================================================")

    noise_profile = create_noise_profile(
        audio,
        noise_rms
    )

    print(
        "Noise profile bins:",
        len(noise_profile)
    )

    # ------------------------------------------------------------
    # Spectral suppression
    # ------------------------------------------------------------

    print("\n================================================")
    print("STEP 6 - SPECTRAL NOISE SUPPRESSION")
    print("================================================")

    spectral_output = (
        spectral_noise_suppression(
            noise_only_output,
            noise_profile,
            strength=1.0
        )
    )

    print(
        "Spectral suppression completed."
    )

    # ------------------------------------------------------------
    # Speech-preserving blend
    # ------------------------------------------------------------

    print("\n================================================")
    print("STEP 7 - SPEECH PRESERVATION")
    print("================================================")

    blended = speech_preserving_blend(
        audio,
        dtcwt_output,
        spectral_output,
        noise_rms
    )

    # ------------------------------------------------------------
    # Remove DC
    # ------------------------------------------------------------

    blended = remove_dc_offset(
        blended
    )

    # ------------------------------------------------------------
    # Restore speech level
    # ------------------------------------------------------------

    print("\n================================================")
    print("STEP 8 - SPEECH LEVEL RESTORATION")
    print("================================================")

    restored = restore_speech_level(
        audio,
        blended,
        noise_rms
    )

    # ------------------------------------------------------------
    # Peak protection
    # ------------------------------------------------------------

    restored = protect_peak(
        restored
    )

    return restored


# ================================================================
# STEP 20
# COMPLETE PIPELINE
# ================================================================

def run_complete_pipeline():

    print("\n")
    print("================================================")
    print(" DTCWT SPEECH DENOISING SYSTEM")
    print("================================================")

    print("\nInput file:")
    print(INPUT_FILE)

    print("\nFinal output:")
    print(OUTPUT_FILE)

    # ------------------------------------------------------------
    # Temporary WAV
    # ------------------------------------------------------------

    temp_directory = tempfile.gettempdir()

    temp_wav = os.path.join(
        temp_directory,
        "project_input_16k.wav"
    )

    try:

        # --------------------------------------------------------
        # 1. Convert input
        # --------------------------------------------------------

        convert_input_to_wav(
            INPUT_FILE,
            temp_wav
        )

        # --------------------------------------------------------
        # 2. Process complete audio
        # --------------------------------------------------------

        denoised = (
            process_complete_audio(
                temp_wav
            )
        )

        # --------------------------------------------------------
        # 3. Save output
        # --------------------------------------------------------

        print("\n================================================")
        print("STEP 9 - SAVE FINAL OUTPUT")
        print("================================================")

        save_output(
            denoised,
            OUTPUT_FILE
        )

        # --------------------------------------------------------
        # 4. Reload saved output for verification
        # --------------------------------------------------------

        final_audio, final_fs = sf.read(
            OUTPUT_FILE,
            dtype="int16"
        )

        final_audio = (
            final_audio.astype(
                np.float64
            )
        )

        original_audio = load_audio(
            temp_wav
        )

        # --------------------------------------------------------
        # 5. Compare
        # --------------------------------------------------------

        compare_signals(
            original_audio,
            final_audio
        )

        print("\n================================================")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("================================================")

        print("\nInput:")
        print(INPUT_FILE)

        print("\nOutput:")
        print(OUTPUT_FILE)

        print(
            "\nOutput sample rate:",
            final_fs,
            "Hz"
        )

        print(
            "Output samples:",
            len(final_audio)
        )

        print(
            "Output duration:",
            round(
                len(final_audio) / final_fs,
                3
            ),
            "seconds"
        )

        print("\nThe final output is a WAV file containing:")
        print("16 kHz")
        print("Mono")
        print("16-bit PCM")
        print("DTCWT-based denoised speech")

    finally:

        # --------------------------------------------------------
        # Delete temporary WAV
        # --------------------------------------------------------

        if os.path.exists(
            temp_wav
        ):

            try:
                os.remove(
                    temp_wav
                )

            except Exception:
                pass


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    run_complete_pipeline()