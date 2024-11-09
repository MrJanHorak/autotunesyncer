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

Clone the repository and install dependencies:

### Frontend Setup

```bash
git clone https://github.com/yourusername/AutoTuneSyncer.git
cd AutoTuneSyncer
npm install
```

### Backend Setup

Navigate to the backend directory and install dependencies:

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
```