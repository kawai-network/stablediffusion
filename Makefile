.PHONY: help test-integration test-integration-ubuntu test-integration-macos test-integration-windows test-integration-all

# Default values
MODEL_URL ?= ""
TEST_TIMEOUT ?= 15

help:
	@echo "Available targets:"
	@echo "  test-integration              - Run integration test on all platforms (manual trigger)"
	@echo "  test-integration-ubuntu       - Run integration test on Ubuntu only"
	@echo "  test-integration-macos        - Run integration test on macOS only"
	@echo "  test-integration-windows      - Run integration test on Windows only"
	@echo ""
	@echo "With custom model:"
	@echo "  make test-integration MODEL_URL=https://example.com/model.gguf"
	@echo "  make test-integration MODEL_URL=https://example.com/model.gguf TEST_TIMEOUT=30"
	@echo ""

test-integration:
	@echo "Triggering integration test on all platforms..."
	@if [ -n "$(MODEL_URL)" ]; then \
		gh workflow run integration-test.yaml -f model_url=$(MODEL_URL) -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test.yaml -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-integration-ubuntu:
	@echo "Triggering integration test on Ubuntu..."
	@if [ -n "$(MODEL_URL)" ]; then \
		gh workflow run integration-test.yaml -f model_url=$(MODEL_URL) -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test.yaml -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-integration-macos:
	@echo "Triggering integration test on macOS..."
	@if [ -n "$(MODEL_URL)" ]; then \
		gh workflow run integration-test.yaml -f model_url=$(MODEL_URL) -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test.yaml -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-integration-windows:
	@echo "Triggering integration test on Windows..."
	@if [ -n "$(MODEL_URL)" ]; then \
		gh workflow run integration-test.yaml -f model_url=$(MODEL_URL) -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test.yaml -f test_timeout=$(TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-integration-all: test-integration-ubuntu test-integration-macos test-integration-windows
	@echo "All integration tests triggered!"
