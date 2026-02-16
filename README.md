# stablediffusion

Go bindings for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - Multi-platform C++ library wrapper.

## Overview

This package provides Go bindings for stable-diffusion.cpp, enabling image and video generation in Go applications. It uses purego for cross-platform dynamic library loading without CGO.

## Features

- Text-to-image generation
- Image-to-image generation
- Video generation
- Model upscaling
- Multi-platform support (Linux, macOS, Windows)
- GPU acceleration (CUDA, ROCm, Vulkan, Metal)
- Pure Go implementation (no CGO required)

## Installation

```bash
go get github.com/kawai-network/stablediffusion
```

## Usage

```go
import "github.com/kawai-network/stablediffusion"

// Load library
sd, err := stablediffusion.New(stablediffusion.LibraryConfig{
    LibPath: "/path/to/libraries",
})
if err != nil {
    log.Fatal(err)
}
defer sd.Close()

// Create context
params := &stablediffusion.SDContextParams{}
stablediffusion.ContextParamsInit(params)
params.ModelPath = stablediffusion.CString("model.safetensors")

ctx, err := sd.NewContext(params)
if err != nil {
    log.Fatal(err)
}
defer ctx.Free()

// Generate image
imgParams := &stablediffusion.SDImgGenParams{}
sd.ImgGenParamsInit(imgParams)
imgParams.Prompt = stablediffusion.CString("a beautiful landscape")
imgParams.Width = 512
imgParams.Height = 512

img := ctx.GenerateImage(imgParams)
```

## Library Loading

The library supports loading from:
- Explicit file path
- Directory with auto-detection (GPU-specific libraries on Windows)

Platform-specific library names:
- Linux: `libstable-diffusion.so`
- macOS: `libstable-diffusion.dylib`
- Windows: `stable-diffusion.dll`

## Building from Source

See [native/](native/) for C++ build instructions.

## Testing

```bash
go test ./...
```

## License

MIT License
