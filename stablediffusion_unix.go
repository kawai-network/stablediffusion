//go:build darwin || linux

package stablediffusion

import (
	"github.com/ebitengine/purego"
)

// openLibrary opens dynamic library - Unix platforms (macOS/Linux)
func openLibrary(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

// closeLibrary closes dynamic library - Unix platforms (macOS/Linux)
func closeLibrary(handle uintptr) error {
	return purego.Dlclose(handle)
}
