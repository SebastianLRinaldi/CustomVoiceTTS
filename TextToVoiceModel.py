from TTS.tts.utils.io import load_checkpoint
from TTS.tts.utils.model import build_model
from TTS.tts.utils.synthesis import synthesis
import soundfile as sf

def text_to_speech(text, output_filename, voice_model_path, voice_config_path):
    # Load your custom voice model
    voice_model = build_model(voice_config_path)
    checkpoint = load_checkpoint(voice_model_path)
    voice_model.load_state_dict(checkpoint["model"])
    voice_model.eval()

    # Generate speech from text
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(
        text,
        model=voice_model,
        config=voice_config_path,
        use_cuda=False  # Set to True if you have a GPU
    )

    # Save the generated speech as an audio file
    sf.write(output_filename, waveform, voice_model.hparams.sampling_rate)

# Usage
text = "Hello, this is a test."
output_filename = "output.wav"
voice_model_path = "path/to/your/voice/model.pth"
voice_config_path = "path/to/your/voice/config.json"

text_to_speech(text, output_filename, voice_model_path, voice_config_path)
