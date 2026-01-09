# Changelog

## [1.0.4] - 2025-05-18

### Added
- **DeepSeek-MoE Support**: Full support for DeepSeek-MoE architectures, including specialized offloading strategies and OOM prevention.
- **SSD Offloading**: Enhanced `OffloadedDynamicKVCache` to support "Flowing" weights and direct disk-to-GPU streaming for large context models.
- **Falcon3 & Dolphin Support**: Added support for Falcon3-MoE and Dolphin-24B (Mistral based) models.
- **Multimodal Support**: Added support for multimodal models like `voxtral-small-24B` (audio) and `gemma3-12B` (image).
- **Qwen3-Next Support**: Added support for the massive 160GB `qwen3-next-80B` model.

### Fixed
- **OOM Prevention**: Implemented `OffloadedDynamicKVCache` to prevent Out Of Memory errors on consumer GPUs by offloading KV cache to SSD.
- **Device-Side Asserts**: Fixed CUDA device-side assert errors in DeepSeek models caused by vocabulary mismatches and invalid `position_ids` slicing.
- **Flash Attention**: Improved fallback mechanisms for systems without `flash-attn` installed.
- **Cache Integrity**: Fixed issues with prompt duplication in `DynamicCache` during pre-fill phases.

### Changed
- **Dependency Updates**: Updated dependencies to support newer Transformers versions.
- **Internal Refactoring**: Refactored internal imports to relative paths for better package portability.
