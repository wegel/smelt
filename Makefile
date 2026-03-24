CONTAINER_RUNTIME ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
DIST_DIR ?= dist
BIN_DIR ?= $(HOME)/.local/bin

.PHONY: build-vulkan build-cuda install-vulkan install-cuda

build-vulkan:
	CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) bash scripts/build-container.sh vulkan $(DIST_DIR)/smelt-vulkan

build-cuda:
	CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) bash scripts/build-container.sh cuda $(DIST_DIR)/smelt-cuda

install-vulkan: build-vulkan
	install -Dm755 $(DIST_DIR)/smelt-vulkan $(BIN_DIR)/smelt-vulkan

install-cuda: build-cuda
	install -Dm755 $(DIST_DIR)/smelt-cuda $(BIN_DIR)/smelt-cuda
