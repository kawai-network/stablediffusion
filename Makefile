.PHONY: help test-integration test-integration-ubuntu test-integration-macos test-integration-windows test-integration-all test-video test-video-all

MODEL_URL ?= ""
TEST_TIMEOUT ?= 15
VIDEO_MODEL_URL ?= ""
VIDEO_TEST_TIMEOUT ?= 30

help:
	@echo "Available targets:"
	@echo "  test-integration              - Run image integration test on all platforms"
	@echo "  test-integration-ubuntu       - Run image integration test on Ubuntu only"
	@echo "  test-integration-macos        - Run image integration test on macOS only"
	@echo "  test-integration-windows      - Run image integration test on Windows only"
	@echo "  test-video                    - Run video integration test on all platforms"
	@echo "  test-video-ubuntu             - Run video integration test on Ubuntu only"
	@echo "  test-video-macos              - Run video integration test on macOS only"
	@echo "  test-video-windows            - Run video integration test on Windows only"
	@echo ""
	@echo "Examples:"
	@echo "  make test-integration"
	@echo "  make test-integration MODEL_URL=https://example.com/model.gguf TEST_TIMEOUT=30"
	@echo "  make test-video VIDEO_MODEL_URL=https://example.com/video-model.gguf VIDEO_TEST_TIMEOUT=60"
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

test-video:
	@echo "Triggering video integration test on all platforms..."
	@if [ -n "$(VIDEO_MODEL_URL)" ]; then \
		gh workflow run integration-test-video.yaml -f model_url=$(VIDEO_MODEL_URL) -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test-video.yaml -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-video-ubuntu:
	@echo "Triggering video integration test on Ubuntu..."
	@if [ -n "$(VIDEO_MODEL_URL)" ]; then \
		gh workflow run integration-test-video.yaml -f model_url=$(VIDEO_MODEL_URL) -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test-video.yaml -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-video-macos:
	@echo "Triggering video integration test on macOS..."
	@if [ -n "$(VIDEO_MODEL_URL)" ]; then \
		gh workflow run integration-test-video.yaml -f model_url=$(VIDEO_MODEL_URL) -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test-video.yaml -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-video-windows:
	@echo "Triggering video integration test on Windows..."
	@if [ -n "$(VIDEO_MODEL_URL)" ]; then \
		gh workflow run integration-test-video.yaml -f model_url=$(VIDEO_MODEL_URL) -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	else \
		gh workflow run integration-test-video.yaml -f test_timeout=$(VIDEO_TEST_TIMEOUT) --repo github.com/kawai-network/stablediffusion; \
	fi

test-video-all: test-video-ubuntu test-video-macos test-video-windows
	@echo "All video integration tests triggered!"
