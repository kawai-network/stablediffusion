package stablediffusion

import (
	"runtime"
	"testing"
)

func TestLibraryName(t *testing.T) {
	name := LibraryName()

	switch runtime.GOOS {
	case "darwin":
		if name != "libstable-diffusion.dylib" {
			t.Errorf("Expected libstable-diffusion.dylib on macOS, got %s", name)
		}
	case "windows":
		if name != "stable-diffusion.dll" {
			t.Errorf("Expected stable-diffusion.dll on Windows, got %s", name)
		}
	default:
		if name != "libstable-diffusion.so" {
			t.Errorf("Expected libstable-diffusion.so on Linux, got %s", name)
		}
	}
}

func TestCString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello", "hello"},
		{"", ""},
		{"test string", "test string"},
	}

	for _, tt := range tests {
		cstr := CString(tt.input)
		if tt.input == "" {
			if cstr != nil {
				t.Errorf("CString(\"\") should return nil, got %v", cstr)
			}
			continue
		}

		result := CGoString(cstr)
		if result != tt.expected {
			t.Errorf("CString/CGoString roundtrip failed: expected %s, got %s", tt.expected, result)
		}
	}
}

func TestCGoStringNil(t *testing.T) {
	result := CGoString(nil)
	if result != "" {
		t.Errorf("CGoString(nil) should return empty string, got %s", result)
	}
}

func TestFindBestLibrary(t *testing.T) {
	// Test with non-existent directory
	result := findBestLibrary("/nonexistent/path", "")
	if result != "" {
		t.Errorf("Expected empty string for non-existent path, got %s", result)
	}
}
