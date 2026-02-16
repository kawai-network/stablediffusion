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
		modelPath = "models/sd-v1-5-q4_1.gguf"
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Fatalf("Model not found at %s - image generation test requires model", modelPath)
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
		// Initialize context parameters
		var params SDContextParams
		sd.ContextParamsInit(&params)

		// Set model path
		params.DiffusionModelPath = CString(modelPath)
		params.NThreads = -1 // Use all available threads
		params.WType = SDTypeQ4_1

		t.Logf("Creating context with model: %s", modelPath)

		// Create context
		ctx, err := sd.NewContext(&params)
		if err != nil {
			t.Fatalf("Failed to create context: %v", err)
		}
		defer ctx.Free()

		t.Log("Context created successfully")
	})

	t.Run("GenerateImage", func(t *testing.T) {
		// Initialize context parameters
		var ctxParams SDContextParams
		sd.ContextParamsInit(&ctxParams)

		ctxParams.DiffusionModelPath = CString(modelPath)
		ctxParams.NThreads = -1
		ctxParams.WType = SDTypeQ4_1

		t.Logf("Creating context with model: %s", modelPath)

		// Create context
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
