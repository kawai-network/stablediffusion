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
}
