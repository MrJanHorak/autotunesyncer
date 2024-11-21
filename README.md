# AutoTuneSyncer

AutoTuneSyncer is an application that synchronizes audio files with MIDI data, enabling seamless music production and editing.

## Description

AutoTuneSyncer is an application that synchronizes audio files with MIDI data, enabling seamless music production and editing.

## Features

- Synchronize audio and MIDI data effortlessly
- Real-time audio processing
- User-friendly interface
- Cross-platform support

## Backend

The backend of AutoTuneSyncer is built with Node.js and Express.js. It handles server-side logic, including audio processing, MIDI data synchronization, and provides API endpoints for the frontend.

### Technologies Used

- **Node.js**: JavaScript runtime environment
- **Express.js**: Web application framework for Node.js
- **MIDI Parser JS**: For parsing MIDI files

## Installation

### Prerequisites

Before installing AutoTuneSyncer, ensure you have the following installed:
- Python 3.8 or higher
- Node.js 14 or higher
- CUDA Toolkit 11.x (for GPU support)
- FFmpeg
- Visual C++ Build Tools (Windows only)

### System Dependencies

1. For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip nodejs ffmpeg
```

2. Navigate to the backend directory and install Node.js dependencies:
```bash
cd backend
npm install
```

## Usage

### Running the Backend

Start the backend server:

```bash
cd backend
npm start
```

### Running the Frontend

In a new terminal window, start the frontend application:

```bash
cd frontend
npm run dev
```

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
MIDI Parser JS for MIDI file parsing
Thanks to all the contributors and the open-source community.

## Reference Links

- [MIDI Parser JS Specifications](https://github.com/colxi/midi-parser-js/wiki/MIDI-File-Format-Specifications)

- [MIDI File Format Specs](https://github.com/colxi/midi-parser-js/wiki/MIDI-File-Format-Specifications)

To run the backend and handle the video sizes start the server using:
```bash
node --max-old-space-size=8192 server.js
````

## Installation

### Prerequisites

Before installing AutoTuneSyncer, ensure you have the following installed:
- Python 3.8 or higher
- Node.js 14 or higher
- CUDA Toolkit 11.x (for GPU support)
- FFmpeg

### Backend Setup

1. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-dev python3-pip
```

2. For Windows:
   - Install [FFmpeg](https://ffmpeg.org/download.html)
   - Add FFmpeg to system PATH
   - Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

3. Install Python dependencies:
```bash
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Install CUDA dependencies (for GPU support):
   - Download and install [CUDA Toolkit 11.x](https://developer.nvidia.com/cuda-downloads)
   - Download and install [cuDNN](https://developer.nvidia.com/cudnn)

### Running the Backend

1. Configure memory settings for video processing:
```bash
# For Linux/Mac
export NODE_OPTIONS="--max-old-space-size=8192"

# For Windows (PowerShell)
$env:NODE_OPTIONS="--max-old-space-size=8192"
```

2. Start the backend server:
```bash
cd backend
python server.py
```

### Memory Management

For processing large videos, you may need to adjust the memory settings:

```bash
# Increase Node.js memory limit
node --max-old-space-size=8192 server.js

# Set FFmpeg memory limit (in environment)
export FFMPEG_MEMORY_LIMIT=256M


### [README.md](file:///c:/Users/janny/development/autotunesyncer/README.md)

Update the installation section with clearer prerequisites and setup instructions.

```markdown


// ...existing code until Installation section...

## Installation

### Prerequisites

Before installing AutoTuneSyncer, ensure you have the following installed:
- Python 3.8 or higher
- Node.js 14 or higher
- CUDA Toolkit 11.x (for GPU support)
- FFmpeg
- Visual C++ Build Tools (Windows only)

### System Dependencies

1. For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-dev python3-pip nodejs npm
```

2. For Windows:
- Install [Python](https://www.python.org/downloads/)
- Install [Node.js](https://nodejs.org/)
- Install [FFmpeg](https://ffmpeg.org/download.html)
- Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Add FFmpeg to system PATH

### Project Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AutoTuneSyncer.git
cd AutoTuneSyncer
```

2. Install Python dependencies:
```bash
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Install Node.js dependencies:
```bash
npm install
```

4. Configure environment:
```bash
# For Linux/Mac
export NODE_OPTIONS="--max-old-space-size=8192"
export FFMPEG_MEMORY_LIMIT=256M

# For Windows (PowerShell)
$env:NODE_OPTIONS="--max-old-space-size=8192"
$env:FFMPEG_MEMORY_LIMIT="256M"
```

