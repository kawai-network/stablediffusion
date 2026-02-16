//go:build windows

package stablediffusion

import (
	"golang.org/x/sys/windows"
)

// openLibrary opens dynamic library - Windows platform
func openLibrary(name string) (uintptr, error) {
	handle, err := windows.LoadLibrary(name)
	return uintptr(handle), err
}

// closeLibrary closes dynamic library - Windows platform
func closeLibrary(handle uintptr) error {
	return windows.FreeLibrary(windows.Handle(handle))
}
