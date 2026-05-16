
# Create GIF for Github

## 1. Install Required Tools

### OBS Studio

Ubuntu / Debian:
```bash
sudo apt install obs-studio
```

Fedora
```bash
sudo dnf install obs-studio
```

### ffmpeg

Ubuntu / Debian:
```bash
sudo apt iinstall ffmpeg
```

Fedora
```bash
sudo dnf install ffmpeg
```

# 2. Record Video with OBS

Open OBS via command: `obs`

Recording video in the application and save the output video

# 3. Covert MP4 to GIF

## Simple Covert

```bash
ffmpeg -i demo.mp4 -vf "fps=15,scale=900:-1:flags=lanczos" demo.gif
```
