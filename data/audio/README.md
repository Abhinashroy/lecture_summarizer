# Audio Files Directory

This directory should contain audio files for testing the lecture summarizer.

## Supported Formats:
- .mp3
- .wav  
- .m4a
- .flac
- .ogg

## Usage:
Place your audio lecture files here and they can be processed through:
1. Web interface: `python main.py --web`
2. Command line: `python main.py --audio_file "data/audio/your_file.mp3"`

## Note:
Large audio files are ignored by git (see .gitignore)