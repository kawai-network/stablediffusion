package stablediffusion

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Check if native library exists
	nativeDir := "native"
	libName := LibraryName()
	libPath := filepath.Join(nativeDir, libName)

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skipf("Native library not found at %s, skipping integration test", libPath)
	}

	// Load library
	sd, err := New(LibraryConfig{
		LibPath: nativeDir,
	})
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer sd.Close()

	// Test basic library functions
	t.Run("GetSystemInfo", func(t *testing.T) {
		info := sd.GetSystemInfo()
		if info == "" {
			t.Error("GetSystemInfo returned empty string")
		}
		t.Logf("System Info: %s", info)
	})

	t.Run("Version", func(t *testing.T) {
		version := sd.Version()
		if version == "" {
			t.Error("Version returned empty string")
		}
		t.Logf("Version: %s", version)
	})

	t.Run("Commit", func(t *testing.T) {
		commit := sd.Commit()
		if commit == "" {
			t.Error("Commit returned empty string")
		}
		t.Logf("Commit: %s", commit)

		// Verify commit hash matches expected version
		expectedCommit := "4ff2c8c"
		if commit != expectedCommit && len(commit) >= len(expectedCommit) {
			actualShort := commit[:len(expectedCommit)]
			if actualShort != expectedCommit {
				t.Errorf("Expected commit to start with %s, got %s", expectedCommit, actualShort)
			}
		}
	})

	t.Run("ContextCreation", func(t *testing.T) {
		// Test that we can initialize context parameters
		var params SDContextParams
		sd.ContextParamsInit(&params)

		// Verify default values are set
		if params.NThreads == 0 {
			t.Log("NThreads initialized to 0 (will use default)")
		}

		t.Log("Context parameters initialized successfully")
	})

	t.Run("SampleParamsInit", func(t *testing.T) {
		// Test sample parameters initialization
		var params SDSampleParams
		sd.SampleParamsInit(&params)

		t.Log("Sample parameters initialized successfully")
	})

	t.Run("ImgGenParamsInit", func(t *testing.T) {
		// Test image generation parameters initialization
		var params SDImgGenParams
		sd.ImgGenParamsInit(&params)

		t.Log("Image generation parameters initialized successfully")
	})
}

func TestImageGeneration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping image generation test in short mode")
	}

	// Check if native library exists
	nativeDir := "native"
	libName := LibraryName()
	libPath := filepath.Join(nativeDir, libName)

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skipf("Native library not found at %s, skipping test", libPath)
	}

	// Check if model exists
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "models/v1-5-pruned-emaonly.safetensors"
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found - image generation test requires a complete model with VAE and CLIP. Provide model_url in workflow input to test image generation.")
	}

	// Load library
	sd, err := New(LibraryConfig{
		LibPath: nativeDir,
	})
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer sd.Close()

	t.Run("CreateContext", func(t *testing.T) {
		var params SDContextParams
		sd.ContextParamsInit(&params)

		params.ModelPath = CString(modelPath)
		params.NThreads = -1
		params.EnableMmap = true

		t.Logf("Creating context with model: %s", modelPath)

		ctx, err := sd.NewContext(&params)
		if err != nil {
			t.Fatalf("Failed to create context: %v", err)
		}
		defer ctx.Free()

		t.Log("Context created successfully")
	})

	t.Run("GenerateImage", func(t *testing.T) {
		var ctxParams SDContextParams
		sd.ContextParamsInit(&ctxParams)

		ctxParams.ModelPath = CString(modelPath)
		ctxParams.NThreads = -1
		ctxParams.EnableMmap = true

		t.Logf("Creating context with model: %s", modelPath)

		ctx, err := sd.NewContext(&ctxParams)
		if err != nil {
			t.Fatalf("Failed to create context: %v", err)
		}
		defer ctx.Free()

		t.Log("Context created successfully")

		// Initialize image generation parameters
		var imgParams SDImgGenParams
		sd.ImgGenParamsInit(&imgParams)

		imgParams.Prompt = CString("a beautiful sunset over the ocean")
		imgParams.NegativePrompt = CString("blurry, bad quality")
		imgParams.Width = 512
		imgParams.Height = 512
		imgParams.Seed = 42
		imgParams.BatchCount = 1

		// Initialize sample parameters
		sd.SampleParamsInit(&imgParams.SampleParams)
		imgParams.SampleParams.SampleMethod = EulerASampleMethod
		imgParams.SampleParams.Scheduler = DiscreteScheduler
		imgParams.SampleParams.SampleSteps = 20      // Standard steps for SD 1.5
		imgParams.SampleParams.Guidance.TxtCfg = 7.5 // Standard CFG for SD 1.5

		t.Logf("Generating 512x512 image with prompt: 'a beautiful sunset over the ocean'")
		t.Logf("Using %d sampling steps with CFG scale %.1f", imgParams.SampleParams.SampleSteps, imgParams.SampleParams.Guidance.TxtCfg)

		// Generate image
		result := ctx.GenerateImage(&imgParams)

		if result == nil {
			t.Fatal("Image generation returned nil")
		}

		if result.Width != 512 || result.Height != 512 {
			t.Errorf("Expected 512x512 image, got %dx%d", result.Width, result.Height)
		}

		if result.Data == nil {
			t.Error("Image data is nil")
		}

		if result.Channel != 3 {
			t.Errorf("Expected 3 channels (RGB), got %d", result.Channel)
		}

		t.Logf("✓ Image generated successfully: %dx%d, %d channels", result.Width, result.Height, result.Channel)
	})
}

func TestVideoGeneration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping video generation test in short mode")
	}

	nativeDir := "native"
	libName := LibraryName()
	libPath := filepath.Join(nativeDir, libName)

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skipf("Native library not found at %s, skipping test", libPath)
	}

	videoModelPath := os.Getenv("VIDEO_MODEL_PATH")
	if videoModelPath == "" {
		videoModelPath = "models/svd-1-5-q4_1.gguf"
	}

	if _, err := os.Stat(videoModelPath); os.IsNotExist(err) {
		t.Skipf("Video model not found at %s, skipping test", videoModelPath)
	}

	sd, err := New(LibraryConfig{
		LibPath: nativeDir,
	})
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer sd.Close()

	t.Run("CreateVideoContext", func(t *testing.T) {
		var params SDContextParams
		sd.ContextParamsInit(&params)

		params.DiffusionModelPath = CString(videoModelPath)
		params.NThreads = -1
		params.WType = SDTypeF32

		t.Logf("Creating video context with model: %s", videoModelPath)

		ctx, err := sd.NewContext(&params)
		if err != nil {
			t.Fatalf("Failed to create video context: %v", err)
		}
		defer ctx.Free()

		t.Log("Video context created successfully")
	})

	t.Run("GenerateVideo", func(t *testing.T) {
		var ctxParams SDContextParams
		sd.ContextParamsInit(&ctxParams)

		ctxParams.DiffusionModelPath = CString(videoModelPath)
		ctxParams.NThreads = -1
		ctxParams.WType = SDTypeF32

		t.Logf("Creating video context with model: %s", videoModelPath)

		ctx, err := sd.NewContext(&ctxParams)
		if err != nil {
			t.Fatalf("Failed to create video context: %v", err)
		}
		defer ctx.Free()

		t.Log("Video context created successfully")

		var vidParams SDVidGenParams
		sd.VidGenParamsInit(&vidParams)

		vidParams.Width = 1024
		vidParams.Height = 576
		vidParams.VideoFrames = 4
		vidParams.Seed = 42

		sd.SampleParamsInit(&vidParams.SampleParams)
		vidParams.SampleParams.SampleMethod = EulerASampleMethod
		vidParams.SampleParams.Scheduler = DiscreteScheduler
		vidParams.SampleParams.SampleSteps = 10

		t.Logf("Generating %d frames video at %dx%d", vidParams.VideoFrames, vidParams.Width, vidParams.Height)

		frames, numFrames := ctx.GenerateVideo(&vidParams)

		if numFrames == 0 {
			t.Fatal("Video generation returned 0 frames")
		}

		if frames == nil {
			t.Fatal("Video generation returned nil frames")
		}

		t.Logf("✓ Video generated successfully: %d frames", numFrames)

		for i, frame := range frames {
			if frame.Data == nil {
				t.Errorf("Frame %d has nil data", i)
			} else {
				t.Logf("Frame %d: %dx%d, %d channels", i, frame.Width, frame.Height, frame.Channel)
			}
		}
	})

	t.Run("GenerateVideoWithInitImage", func(t *testing.T) {
		var ctxParams SDContextParams
		sd.ContextParamsInit(&ctxParams)

		ctxParams.DiffusionModelPath = CString(videoModelPath)
		ctxParams.NThreads = -1
		ctxParams.WType = SDTypeF32

		ctx, err := sd.NewContext(&ctxParams)
		if err != nil {
			t.Fatalf("Failed to create video context: %v", err)
		}
		defer ctx.Free()

		initImage := SDImage{
			Width:   1024,
			Height:  576,
			Channel: 3,
			Data:    nil,
		}

		var vidParams SDVidGenParams
		sd.VidGenParamsInit(&vidParams)

		vidParams.InitImage = initImage
		vidParams.Width = 1024
		vidParams.Height = 576
		vidParams.VideoFrames = 4
		vidParams.Seed = 123
		vidParams.Strength = 1.0

		sd.SampleParamsInit(&vidParams.SampleParams)
		vidParams.SampleParams.SampleMethod = EulerASampleMethod
		vidParams.SampleParams.Scheduler = DiscreteScheduler
		vidParams.SampleParams.SampleSteps = 10

		t.Logf("Generating video with init image params")

		_, numFrames := ctx.GenerateVideo(&vidParams)

		t.Logf("Video with init image generated: %d frames", numFrames)
	})
}

func TestWANVideoGenerationSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping WAN smoke test in short mode")
	}

	nativeDir := "native"
	libName := LibraryName()
	libPath := filepath.Join(nativeDir, libName)
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skipf("Native library not found at %s, skipping test", libPath)
	}

	wanModelPath := os.Getenv("WAN_MODEL_PATH")
	wanVAEPath := os.Getenv("WAN_VAE_PATH")
	wanT5Path := os.Getenv("WAN_T5_PATH")
	wanHighNoiseModelPath := os.Getenv("WAN_HIGH_NOISE_MODEL_PATH")
	wanClipVisionPath := os.Getenv("WAN_CLIP_VISION_PATH")

	if wanModelPath == "" || wanVAEPath == "" || wanT5Path == "" {
		t.Skip("WAN smoke test requires WAN_MODEL_PATH, WAN_VAE_PATH, and WAN_T5_PATH")
	}

	requiredPaths := map[string]string{
		"WAN_MODEL_PATH": wanModelPath,
		"WAN_VAE_PATH":   wanVAEPath,
		"WAN_T5_PATH":    wanT5Path,
	}

	for env, path := range requiredPaths {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Skipf("%s file not found at %s, skipping test", env, path)
		}
	}

	if wanHighNoiseModelPath != "" {
		if _, err := os.Stat(wanHighNoiseModelPath); os.IsNotExist(err) {
			t.Skipf("WAN_HIGH_NOISE_MODEL_PATH file not found at %s, skipping test", wanHighNoiseModelPath)
		}
	}

	if wanClipVisionPath != "" {
		if _, err := os.Stat(wanClipVisionPath); os.IsNotExist(err) {
			t.Skipf("WAN_CLIP_VISION_PATH file not found at %s, skipping test", wanClipVisionPath)
		}
	}

	sd, err := New(LibraryConfig{
		LibPath: nativeDir,
	})
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer sd.Close()

	var ctxParams SDContextParams
	sd.ContextParamsInit(&ctxParams)

	ctxParams.DiffusionModelPath = CString(wanModelPath)
	ctxParams.HighNoiseDiffusionModelPath = CString(wanHighNoiseModelPath)
	ctxParams.VAEPath = CString(wanVAEPath)
	ctxParams.T5XXLPath = CString(wanT5Path)
	ctxParams.ClipVisionPath = CString(wanClipVisionPath)
	ctxParams.NThreads = -1
	ctxParams.EnableMmap = true
	ctxParams.DiffusionFlashAttn = true
	ctxParams.FlowShift = 3.0

	t.Logf("Creating WAN context with diffusion model: %s", wanModelPath)

	ctx, err := sd.NewContext(&ctxParams)
	if err != nil {
		t.Fatalf("Failed to create WAN context: %v", err)
	}
	defer ctx.Free()

	var vidParams SDVidGenParams
	sd.VidGenParamsInit(&vidParams)

	vidParams.Prompt = CString("a lovely cat")
	vidParams.NegativePrompt = CString("blurry, low quality")
	vidParams.Width = 832
	vidParams.Height = 480
	vidParams.VideoFrames = 2
	vidParams.Seed = 42

	sd.SampleParamsInit(&vidParams.SampleParams)
	vidParams.SampleParams.SampleMethod = EulerSampleMethod
	vidParams.SampleParams.Scheduler = DiscreteScheduler
	vidParams.SampleParams.SampleSteps = 8
	vidParams.SampleParams.Guidance.TxtCfg = 6.0

	if wanHighNoiseModelPath != "" {
		sd.SampleParamsInit(&vidParams.HighNoiseSampleParams)
		vidParams.HighNoiseSampleParams.SampleMethod = EulerSampleMethod
		vidParams.HighNoiseSampleParams.Scheduler = DiscreteScheduler
		vidParams.HighNoiseSampleParams.SampleSteps = 8
		vidParams.HighNoiseSampleParams.Guidance.TxtCfg = 3.5
	}

	t.Logf("Generating WAN smoke video: %d frames at %dx%d", vidParams.VideoFrames, vidParams.Width, vidParams.Height)

	frames, numFrames := ctx.GenerateVideo(&vidParams)
	if numFrames == 0 {
		t.Fatal("WAN video generation returned 0 frames")
	}
	if frames == nil {
		t.Fatal("WAN video generation returned nil frames")
	}
	if frames[0].Data == nil {
		t.Fatal("WAN video generation returned frame with nil data")
	}

	t.Logf("✓ WAN smoke generation succeeded: %d frames", numFrames)
}
