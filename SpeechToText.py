import whisper
import sounddevice as sd # Needed to avoid ALSA Errors
from speech_recognition import (
    Recognizer, 
    AudioData, 
    Microphone, 
    UnknownValueError, 
    RequestError,
)

from numpy.typing import NDArray
from numpy import (
    int16 as npInt16,
    float32 as npFloat32,
    frombuffer as npFromBuffer,
)

from piper.voice import PiperVoice

# Whisper - STT
whisper_model = "./ml/whisper_models/base.pt"
stt = whisper.load_model(whisper_model)  # Choose your desired Whisper model
recognizer = Recognizer()

# Piper - TTS
piper_model = "./ml/piper_models/en_US-ryan-high.onnx"
tts = PiperVoice.load(piper_model)

# Find the index of your microphone
def get_microphone_index() -> int:
    EXPECTED_MIC_NAME : str = "Logitech USB Headset"

    for index, name in enumerate(Microphone.list_microphone_names()):
        if EXPECTED_MIC_NAME in name:
            return index

    raise Exception("Microphone not found")

def get_audio_numpy_chunk( mic_device_idx : int) -> NDArray:
    RATE : int = 16000
    CHUNK : int = 1024
    with Microphone(device_index=mic_device_idx, sample_rate=RATE, chunk_size=CHUNK) as source :
        #recognizer.adjust_for_ambient_noise(source)
        #print("\n\nListening using Mic Idx [", mic_device_idx, "]...")
        audio_data : AudioData = recognizer.listen(source, stream=False)

    audio_np_data = npFromBuffer(buffer=audio_data.get_wav_data(), dtype=npInt16).flatten().astype(npFloat32) / 32768.0
    #print("Have Audio...")

    return audio_np_data

def transcribe(audio_numpy_data : NDArray) -> dict[str, str | list]:
    try:
        #print("Transcribing...")
        text = stt.transcribe(audio=audio_numpy_data, language='en', fp16=False)
        return text

    except UnknownValueError:
        print("Whisper could not understand audio")
    except RequestError as e:
        print("Could not request results from Whisper service; {0}".format(e))

def speak( text : str) :
    # Setup a sounddevice OutputStream with appropriate parameters
    # The sample rate and channels should match the properties of the PCM data
    stream = sd.OutputStream(samplerate=tts.config.sample_rate, channels=1, dtype='int16')
    stream.start()
    for audio_bytes in tts.synthesize_stream_raw(text):
        int_data = npFromBuffer(audio_bytes, dtype=npInt16)
        stream.write(int_data)
    stream.stop()
    stream.close()

# Main function to record and transcribe
if __name__ == "__main__":
    microphone_index : int = get_microphone_index()
    
    try:
        while True:
            audio_data : NDArray = get_audio_numpy_chunk(mic_device_idx=microphone_index)
            text_data : dict[str, str | list] = transcribe(audio_numpy_data=audio_data)
            print("Text: " + text_data["text"])
            speak( text_data["text"] )

    except KeyboardInterrupt:
        print("Stopped streaming audio.")
