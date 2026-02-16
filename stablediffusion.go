package stablediffusion

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Define enum types
type RngType int32

const (
	DefaultRNG RngType = iota
	CUDARNG
	CPURNG
	RNGTypeCount
)

type SampleMethod int32

const (
	EulerSampleMethod SampleMethod = iota
	EulerASampleMethod
	HeunSampleMethod
	DPM2SampleMethod
	DPMPP2SASampleMethod
	DPMPP2MSampleMethod
	DPMPP2Mv2SampleMethod
	IPNDMSampleMethod
	IPNDMSampleMethodV
	LCMSampleMethod
	DDIMTrailingSampleMethod
	TCDSampleMethod
	SampleMethodCount
)

type Scheduler int32

const (
	DiscreteScheduler Scheduler = iota
	KarrasScheduler
	ExponentialScheduler
	AYSScheduler
	GITScheduler
	SGMUniformScheduler
	SimpleScheduler
	SmoothstepScheduler
	KLOptimalScheduler
	LCMScheduler
	SchedulerCount
)

type Prediction int32

const (
	EPSPred Prediction = iota
	VPred
	EDMVPred
	FlowPred
	FluxFlowPred
	Flux2FlowPred
	PredictionCount
)

type SDType int32

const (
	SDTypeF32 SDType = iota
	SDTypeF16
	SDTypeQ4_0
	SDTypeQ4_1
	SDTypeQ5_0    = 6
	SDTypeQ5_1    = 7
	SDTypeQ8_0    = 8
	SDTypeQ8_1    = 9
	SDTypeQ2_K    = 10
	SDTypeQ3_K    = 11
	SDTypeQ4_K    = 12
	SDTypeQ5_K    = 13
	SDTypeQ6_K    = 14
	SDTypeQ8_K    = 15
	SDTypeIQ2_XXS = 16
	SDTypeIQ2_XS  = 17
	SDTypeIQ3_XXS = 18
	SDTypeIQ1_S   = 19
	SDTypeIQ4_NL  = 20
	SDTypeIQ3_S   = 21
	SDTypeIQ2_S   = 22
	SDTypeIQ4_XS  = 23
	SDTypeI8      = 24
	SDTypeI16     = 25
	SDTypeI32     = 26
	SDTypeI64     = 27
	SDTypeF64     = 28
	SDTypeIQ1_M   = 29
	SDTypeBF16    = 30
	SDTypeTQ1_0   = 34
	SDTypeTQ2_0   = 35
	SDTypeMXFP4   = 39
	SDTypeCount   = 40
)

type SDLogLevel int32

const (
	SDLogDebug SDLogLevel = iota
	SDLogInfo
	SDLogWarn
	SDLogError
)

type Preview int32

const (
	PreviewNone Preview = iota
	PreviewProj
	PreviewTAE
	PreviewVAE
	PreviewCount
)

type LoraApplyMode int32

const (
	LoraApplyAuto LoraApplyMode = iota
	LoraApplyImmediately
	LoraApplyAtRuntime
	LoraApplyModeCount
)

type SDCacheMode int32

const (
	SDCacheDisabled SDCacheMode = iota
	SDCacheEasycache
	SDCacheUcache
	SDCacheDbcache
	SDCacheTaylorseer
	SDCacheCacheDit
)

// Define structs
type SDTilingParams struct {
	Enabled       bool
	TileSizeX     int32
	TileSizeY     int32
	TargetOverlap float32
	RelSizeX      float32
	RelSizeY      float32
}

type SDEmbedding struct {
	Name *uint8
	Path *uint8
}

type SDContextParams struct {
	ModelPath                   *uint8
	ClipLPath                   *uint8
	ClipGPath                   *uint8
	ClipVisionPath              *uint8
	T5XXLPath                   *uint8
	LLMPath                     *uint8
	LLMVisionPath               *uint8
	DiffusionModelPath          *uint8
	HighNoiseDiffusionModelPath *uint8
	VAEPath                     *uint8
	TAESDPath                   *uint8
	ControlNetPath              *uint8
	Embeddings                  *SDEmbedding
	EmbeddingCount              uint32
	PhotoMakerPath              *uint8
	TensorTypeRules             *uint8
	VAEDecodeOnly               bool
	FreeParamsImmediately       bool
	NThreads                    int32
	WType                       SDType
	RNGType                     RngType
	SamplerRNGType              RngType
	Prediction                  Prediction
	LoraApplyMode               LoraApplyMode
	OffloadParamsToCPU          bool
	EnableMmap                  bool
	KeepClipOnCPU               bool
	KeepControlNetOnCPU         bool
	KeepVAEOnCPU                bool
	DiffusionFlashAttn          bool
	TAEPreviewOnly              bool
	DiffusionConvDirect         bool
	VAEConvDirect               bool
	CircularX                   bool
	CircularY                   bool
	ForceSDXLVAConvScale        bool
	ChromaUseDitMask            bool
	ChromaUseT5Mask             bool
	ChromaT5MaskPad             int32
	QwenImageZeroCondT          bool
	FlowShift                   float32
}

type SDImage struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    *uint8
}

type SDSLGParams struct {
	Layers     *int32
	LayerCount uintptr
	LayerStart float32
	LayerEnd   float32
	Scale      float32
}

type SDGuidanceParams struct {
	TxtCfg            float32
	ImgCfg            float32
	DistilledGuidance float32
	SLG               SDSLGParams
}

type SDSampleParams struct {
	Guidance          SDGuidanceParams
	Scheduler         Scheduler
	SampleMethod      SampleMethod
	SampleSteps       int32
	Eta               float32
	ShiftedTimestep   int32
	CustomSigmas      *float32
	CustomSigmasCount int32
}

type SDPMParams struct {
	IDImages      *SDImage
	IDImagesCount int32
	IDEmbedPath   *uint8
	StyleStrength float32
}

type SDCacheParams struct {
	Mode                     SDCacheMode
	ReuseThreshold           float32
	StartPercent             float32
	EndPercent               float32
	ErrorDecayRate           float32
	UseRelativeThreshold     bool
	ResetErrorOnCompute      bool
	FnComputeBlocks          int32
	BnComputeBlocks          int32
	ResidualDiffThreshold    float32
	MaxWarmupSteps           int32
	MaxCachedSteps           int32
	MaxContinuousCachedSteps int32
	TaylorseerNDerivatives   int32
	TaylorseerSkipInterval   int32
	ScmMask                  *uint8
	ScmPolicyDynamic         bool
}

type SDLora struct {
	IsHighNoise bool
	Multiplier  float32
	Path        *uint8
}

type SDImgGenParams struct {
	Loras              *SDLora
	LoraCount          uint32
	Prompt             *uint8
	NegativePrompt     *uint8
	ClipSkip           int32
	InitImage          SDImage
	RefImages          *SDImage
	RefImagesCount     int32
	AutoResizeRefImage bool
	IncreaseRefIndex   bool
	MaskImage          SDImage
	Width              int32
	Height             int32
	SampleParams       SDSampleParams
	Strength           float32
	Seed               int64
	BatchCount         int32
	ControlImage       SDImage
	ControlStrength    float32
	PMParams           SDPMParams
	VAETilingParams    SDTilingParams
	Cache              SDCacheParams
}

type SDVidGenParams struct {
	Loras                 *SDLora
	LoraCount             uint32
	Prompt                *uint8
	NegativePrompt        *uint8
	ClipSkip              int32
	InitImage             SDImage
	EndImage              SDImage
	ControlFrames         *SDImage
	ControlFramesSize     int32
	Width                 int32
	Height                int32
	SampleParams          SDSampleParams
	HighNoiseSampleParams SDSampleParams
	MOEBoundary           float32
	Strength              float32
	Seed                  int64
	VideoFrames           int32
	VaceStrength          float32
	Cache                 SDCacheParams
}

// Define context types
type SDContext struct {
	ptr unsafe.Pointer
	sd  *StableDiffusion
}

type UpscalerContext struct {
	ptr unsafe.Pointer
	sd  *StableDiffusion
}

// Define callback function types
type SDLogCallback func(level SDLogLevel, text *uint8, data unsafe.Pointer)
type SDProgressCallback func(step int32, steps int32, time float32, data unsafe.Pointer)
type SDPreviewCallback func(step int32, frameCount int32, frames *SDImage, isNoisy bool, data unsafe.Pointer)

// StableDiffusion is the main library handle
type StableDiffusion struct {
	handle uintptr

	// Function pointers
	sdSetLogCallback         func(cb SDLogCallback, data unsafe.Pointer)
	sdSetProgressCallback    func(cb SDProgressCallback, data unsafe.Pointer)
	sdSetPreviewCallback     func(cb SDPreviewCallback, mode Preview, interval int32, denoised bool, noisy bool, data unsafe.Pointer)
	sdGetNumPhysicalCores    func() int32
	sdGetSystemInfo          func() *uint8
	sdTypeName               func(typ SDType) *uint8
	strToSDType              func(str *uint8) SDType
	sdRngTypeName            func(rngType RngType) *uint8
	strToRngType             func(str *uint8) RngType
	sdSampleMethodName       func(method SampleMethod) *uint8
	strToSampleMethod        func(str *uint8) SampleMethod
	sdSchedulerName          func(scheduler Scheduler) *uint8
	strToScheduler           func(str *uint8) Scheduler
	sdPredictionName         func(prediction Prediction) *uint8
	strToPrediction          func(str *uint8) Prediction
	sdPreviewName            func(preview Preview) *uint8
	strToPreview             func(str *uint8) Preview
	sdLoraApplyModeName      func(mode LoraApplyMode) *uint8
	strToLoraApplyMode       func(str *uint8) LoraApplyMode
	sdCacheParamsInit        func(params *SDCacheParams)
	sdContextParamsInit      func(params *SDContextParams)
	sdContextParamsToStr     func(params *SDContextParams) *uint8
	newSDContext             func(params *SDContextParams) unsafe.Pointer
	freeSDContext            func(ctx unsafe.Pointer)
	sdSampleParamsInit       func(params *SDSampleParams)
	sdSampleParamsToStr      func(params *SDSampleParams) *uint8
	sdGetDefaultSampleMethod func(ctx unsafe.Pointer) SampleMethod
	sdGetDefaultScheduler    func(ctx unsafe.Pointer, sampleMethod SampleMethod) Scheduler
	sdImgGenParamsInit       func(params *SDImgGenParams)
	sdImgGenParamsToStr      func(params *SDImgGenParams) *uint8
	generateImage            func(ctx unsafe.Pointer, params *SDImgGenParams) *SDImage
	sdVidGenParamsInit       func(params *SDVidGenParams)
	generateVideo            func(ctx unsafe.Pointer, params *SDVidGenParams, numFramesOut *int32) *SDImage
	newUpscalerContext       func(esrganPath *uint8, offloadParamsToCPU bool, direct bool, nThreads int32, tileSize int32) unsafe.Pointer
	freeUpscalerContext      func(ctx unsafe.Pointer)
	upscale                  func(ctx unsafe.Pointer, inputImage *SDImage, upscaleFactor uint32) *SDImage
	getUpscaleFactor         func(ctx unsafe.Pointer) int32
	convert                  func(inputPath *uint8, vaePath *uint8, outputPath *uint8, outputType SDType, tensorTypeRules *uint8, convertName bool) bool
	preprocessCanny          func(image *SDImage, highThreshold float32, lowThreshold float32, weak float32, strong float32, inverse bool) bool
	sdCommit                 func() *uint8
	sdVersion                func() *uint8
}

// LibraryConfig configures library loading
type LibraryConfig struct {
	LibPath string
	GPUType string
}

// New creates a new StableDiffusion instance with library loading
func New(config LibraryConfig) (*StableDiffusion, error) {
	libPath := config.LibPath
	if libPath == "" {
		libPath = "."
	}

	info, err := os.Stat(libPath)
	if err != nil {
		return nil, fmt.Errorf("invalid library path: %w", err)
	}

	var path string
	if info.IsDir() {
		path = findBestLibrary(libPath, config.GPUType)
		if path == "" {
			return nil, fmt.Errorf("no suitable stable-diffusion library found in %s", libPath)
		}
	} else {
		path = libPath
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %w", err)
	}

	handle, err := openLibrary(absPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load library from %s: %w", absPath, err)
	}

	sd := &StableDiffusion{handle: handle}
	if err := sd.registerFunctions(); err != nil {
		return nil, err
	}

	return sd, nil
}

// Close closes the library
func (sd *StableDiffusion) Close() error {
	if sd.handle != 0 {
		return closeLibrary(sd.handle)
	}
	return nil
}

func (sd *StableDiffusion) registerFunctions() error {
	purego.RegisterLibFunc(&sd.sdSetLogCallback, sd.handle, "sd_set_log_callback")
	purego.RegisterLibFunc(&sd.sdSetProgressCallback, sd.handle, "sd_set_progress_callback")
	purego.RegisterLibFunc(&sd.sdSetPreviewCallback, sd.handle, "sd_set_preview_callback")
	purego.RegisterLibFunc(&sd.sdGetNumPhysicalCores, sd.handle, "sd_get_num_physical_cores")
	purego.RegisterLibFunc(&sd.sdGetSystemInfo, sd.handle, "sd_get_system_info")
	purego.RegisterLibFunc(&sd.sdTypeName, sd.handle, "sd_type_name")
	purego.RegisterLibFunc(&sd.strToSDType, sd.handle, "str_to_sd_type")
	purego.RegisterLibFunc(&sd.sdRngTypeName, sd.handle, "sd_rng_type_name")
	purego.RegisterLibFunc(&sd.strToRngType, sd.handle, "str_to_rng_type")
	purego.RegisterLibFunc(&sd.sdSampleMethodName, sd.handle, "sd_sample_method_name")
	purego.RegisterLibFunc(&sd.strToSampleMethod, sd.handle, "str_to_sample_method")
	purego.RegisterLibFunc(&sd.sdSchedulerName, sd.handle, "sd_scheduler_name")
	purego.RegisterLibFunc(&sd.strToScheduler, sd.handle, "str_to_scheduler")
	purego.RegisterLibFunc(&sd.sdPredictionName, sd.handle, "sd_prediction_name")
	purego.RegisterLibFunc(&sd.strToPrediction, sd.handle, "str_to_prediction")
	purego.RegisterLibFunc(&sd.sdPreviewName, sd.handle, "sd_preview_name")
	purego.RegisterLibFunc(&sd.strToPreview, sd.handle, "str_to_preview")
	purego.RegisterLibFunc(&sd.sdLoraApplyModeName, sd.handle, "sd_lora_apply_mode_name")
	purego.RegisterLibFunc(&sd.strToLoraApplyMode, sd.handle, "str_to_lora_apply_mode")
	purego.RegisterLibFunc(&sd.sdCacheParamsInit, sd.handle, "sd_cache_params_init")
	purego.RegisterLibFunc(&sd.sdContextParamsInit, sd.handle, "sd_ctx_params_init")
	purego.RegisterLibFunc(&sd.sdContextParamsToStr, sd.handle, "sd_ctx_params_to_str")
	purego.RegisterLibFunc(&sd.newSDContext, sd.handle, "new_sd_ctx")
	purego.RegisterLibFunc(&sd.freeSDContext, sd.handle, "free_sd_ctx")
	purego.RegisterLibFunc(&sd.sdSampleParamsInit, sd.handle, "sd_sample_params_init")
	purego.RegisterLibFunc(&sd.sdSampleParamsToStr, sd.handle, "sd_sample_params_to_str")
	purego.RegisterLibFunc(&sd.sdGetDefaultSampleMethod, sd.handle, "sd_get_default_sample_method")
	purego.RegisterLibFunc(&sd.sdGetDefaultScheduler, sd.handle, "sd_get_default_scheduler")
	purego.RegisterLibFunc(&sd.sdImgGenParamsInit, sd.handle, "sd_img_gen_params_init")
	purego.RegisterLibFunc(&sd.sdImgGenParamsToStr, sd.handle, "sd_img_gen_params_to_str")
	purego.RegisterLibFunc(&sd.generateImage, sd.handle, "generate_image")
	purego.RegisterLibFunc(&sd.sdVidGenParamsInit, sd.handle, "sd_vid_gen_params_init")
	purego.RegisterLibFunc(&sd.generateVideo, sd.handle, "generate_video")
	purego.RegisterLibFunc(&sd.newUpscalerContext, sd.handle, "new_upscaler_ctx")
	purego.RegisterLibFunc(&sd.freeUpscalerContext, sd.handle, "free_upscaler_ctx")
	purego.RegisterLibFunc(&sd.upscale, sd.handle, "upscale")
	purego.RegisterLibFunc(&sd.getUpscaleFactor, sd.handle, "get_upscale_factor")
	purego.RegisterLibFunc(&sd.convert, sd.handle, "convert")
	purego.RegisterLibFunc(&sd.preprocessCanny, sd.handle, "preprocess_canny")
	purego.RegisterLibFunc(&sd.sdCommit, sd.handle, "sd_commit")
	purego.RegisterLibFunc(&sd.sdVersion, sd.handle, "sd_version")
	return nil
}

func findBestLibrary(dir string, gpuType string) string {
	ext := ".so"
	prefix := "lib"

	switch runtime.GOOS {
	case "darwin":
		ext = ".dylib"
	case "windows":
		ext = ".dll"
		prefix = ""
	}

	if runtime.GOOS == "windows" && gpuType != "" {
		switch strings.ToUpper(gpuType) {
		case "NVIDIA":
			if path := checkLib(dir, "cuda12", prefix, ext); path != "" {
				return path
			}
		case "AMD":
			if path := checkLib(dir, "rocm", prefix, ext); path != "" {
				return path
			}
		case "VULKAN":
			if path := checkLib(dir, "vulkan", prefix, ext); path != "" {
				return path
			}
		}
	}

	return checkLib(dir, "", prefix, ext)
}

func checkLib(dir, subdir, prefix, ext string) string {
	var libPath string
	if subdir != "" {
		libPath = filepath.Join(dir, subdir, prefix+"stable-diffusion"+ext)
	} else {
		libPath = filepath.Join(dir, prefix+"stable-diffusion"+ext)
	}

	if _, err := os.Stat(libPath); err == nil {
		return libPath
	}
	return ""
}

// LibraryName returns platform-specific library name
func LibraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libstable-diffusion.dylib"
	case "windows":
		return "stable-diffusion.dll"
	default:
		return "libstable-diffusion.so"
	}
}

// Helper functions
func CGoString(cStr *uint8) string {
	if cStr == nil {
		return ""
	}
	var len int
	for p := cStr; *p != 0; p = (*uint8)(unsafe.Add(unsafe.Pointer(p), 1)) {
		len++
	}
	return string(unsafe.Slice(cStr, len))
}

func CString(str string) *uint8 {
	if str == "" {
		return nil
	}
	buf := make([]uint8, len(str)+1)
	copy(buf, str)
	buf[len(str)] = 0
	return &buf[0]
}

func FreeCString(cStr *uint8) {
	// No-op in Go - memory managed by GC
}

// StableDiffusion methods
func (sd *StableDiffusion) ContextParamsInit(params *SDContextParams) {
	sd.sdContextParamsInit(params)
}

func (sd *StableDiffusion) NewContext(params *SDContextParams) (*SDContext, error) {
	ptr := sd.newSDContext(params)
	if ptr == nil {
		return nil, fmt.Errorf("failed to create SD context")
	}
	return &SDContext{ptr: ptr, sd: sd}, nil
}

func (ctx *SDContext) Free() {
	if ctx.ptr != nil {
		ctx.sd.freeSDContext(ctx.ptr)
		ctx.ptr = nil
	}
}

func (sd *StableDiffusion) SampleParamsInit(params *SDSampleParams) {
	sd.sdSampleParamsInit(params)
}

func (sd *StableDiffusion) ImgGenParamsInit(params *SDImgGenParams) {
	sd.sdImgGenParamsInit(params)
}

// SetProgressCallback sets the progress callback function
func (sd *StableDiffusion) SetProgressCallback(cb func(step int, steps int, time float32, data interface{}), data interface{}) {
	if cb == nil {
		sd.sdSetProgressCallback(nil, nil)
		return
	}

	cCallback := func(step int32, steps int32, time float32, cData unsafe.Pointer) {
		cb(int(step), int(steps), time, data)
	}

	sd.sdSetProgressCallback(cCallback, nil)
}

func (ctx *SDContext) GenerateImage(params *SDImgGenParams) *SDImage {
	return ctx.sd.generateImage(ctx.ptr, params)
}

func (sd *StableDiffusion) VidGenParamsInit(params *SDVidGenParams) {
	sd.sdVidGenParamsInit(params)
}

func (ctx *SDContext) GenerateVideo(params *SDVidGenParams) ([]SDImage, int) {
	var numFrames int32
	framesPtr := ctx.sd.generateVideo(ctx.ptr, params, &numFrames)
	if framesPtr == nil {
		return nil, 0
	}

	frames := make([]SDImage, numFrames)
	for i := range frames {
		frames[i] = *(*SDImage)(unsafe.Add(unsafe.Pointer(framesPtr), uintptr(i)*unsafe.Sizeof(SDImage{})))
	}
	return frames, int(numFrames)
}

func (sd *StableDiffusion) GetSystemInfo() string {
	return CGoString(sd.sdGetSystemInfo())
}

func (sd *StableDiffusion) Version() string {
	return CGoString(sd.sdVersion())
}

func (sd *StableDiffusion) Commit() string {
	return CGoString(sd.sdCommit())
}

// GetDefaultSampleMethod gets the default sample method for the context
func (sd *StableDiffusion) GetDefaultSampleMethod(ctx *SDContext) SampleMethod {
	return sd.sdGetDefaultSampleMethod(ctx.ptr)
}

// GetDefaultScheduler gets the default scheduler for the context and sample method
func (sd *StableDiffusion) GetDefaultScheduler(ctx *SDContext, sampleMethod SampleMethod) Scheduler {
	return sd.sdGetDefaultScheduler(ctx.ptr, sampleMethod)
}

// NewUpscalerContext creates a new upscaler context
func (sd *StableDiffusion) NewUpscalerContext(esrganPath string, offloadParamsToCPU bool, direct bool, nThreads int32, tileSize int32) (*UpscalerContext, error) {
	cPath := CString(esrganPath)
	defer FreeCString(cPath)

	ptr := sd.newUpscalerContext(cPath, offloadParamsToCPU, direct, nThreads, tileSize)
	if ptr == nil {
		return nil, fmt.Errorf("failed to create upscaler context")
	}
	return &UpscalerContext{ptr: ptr, sd: sd}, nil
}

// CacheParamsInit initializes cache parameters
func (sd *StableDiffusion) CacheParamsInit(params *SDCacheParams) {
	sd.sdCacheParamsInit(params)
}

// PreprocessCanny preprocesses image with Canny edge detection
func (sd *StableDiffusion) PreprocessCanny(image SDImage, highThreshold, lowThreshold, weak, strong float32, inverse bool) bool {
	return sd.preprocessCanny(&image, highThreshold, lowThreshold, weak, strong, inverse)
}

// Convenience variables for package-level access (requires library to be loaded)
// These are shortcuts that delegate to a default instance once initialized

var defaultSD *StableDiffusion

// SetDefaultInstance sets the default StableDiffusion instance for package-level functions
func SetDefaultInstance(sd *StableDiffusion) {
	defaultSD = sd
}

// ContextParamsInit initializes context parameters using default instance
func ContextParamsInit(params *SDContextParams) {
	if defaultSD != nil {
		defaultSD.ContextParamsInit(params)
	}
}

// NewContext creates a new context using default instance
func NewContext(params *SDContextParams) (*SDContext, error) {
	if defaultSD == nil {
		return nil, fmt.Errorf("no default StableDiffusion instance set")
	}
	return defaultSD.NewContext(params)
}

// ImgGenParamsInit initializes image generation parameters using default instance
func ImgGenParamsInit(params *SDImgGenParams) {
	if defaultSD != nil {
		defaultSD.ImgGenParamsInit(params)
	}
}

// VidGenParamsInit initializes video generation parameters using default instance
func VidGenParamsInit(params *SDVidGenParams) {
	if defaultSD != nil {
		defaultSD.VidGenParamsInit(params)
	}
}

// SampleParamsInit initializes sample parameters using default instance
func SampleParamsInit(params *SDSampleParams) {
	if defaultSD != nil {
		defaultSD.SampleParamsInit(params)
	}
}

// CacheParamsInit initializes cache parameters using default instance
func CacheParamsInit(params *SDCacheParams) {
	if defaultSD != nil {
		defaultSD.sdCacheParamsInit(params)
	}
}

// GenerateImage generates image using default instance
func GenerateImage(params *SDImgGenParams) *SDImage {
	if defaultSD == nil {
		return nil
	}
	// Note: This requires a context, so users should use the instance method directly
	return nil
}

// PreprocessCanny preprocesses image with Canny edge detection using default instance
func PreprocessCanny(image SDImage, highThreshold, lowThreshold, weak, strong float32, inverse bool) bool {
	if defaultSD == nil {
		return false
	}
	return defaultSD.preprocessCanny(&image, highThreshold, lowThreshold, weak, strong, inverse)
}

// Convert converts model using default instance
func Convert(inputPath, vaePath, outputPath string, outputType SDType, tensorTypeRules string, convertName bool) (bool, error) {
	if defaultSD == nil {
		return false, fmt.Errorf("no default StableDiffusion instance set")
	}
	cInputPath := CString(inputPath)
	cVaePath := CString(vaePath)
	cOutputPath := CString(outputPath)
	cTensorTypeRules := CString(tensorTypeRules)

	defer func() {
		FreeCString(cInputPath)
		FreeCString(cVaePath)
		FreeCString(cOutputPath)
		FreeCString(cTensorTypeRules)
	}()

	return defaultSD.convert(cInputPath, cVaePath, cOutputPath, outputType, cTensorTypeRules, convertName), nil
}

// UpscalerContext methods

// Free frees the upscaler context
func (ctx *UpscalerContext) Free() {
	if ctx.ptr != nil {
		ctx.sd.freeUpscalerContext(ctx.ptr)
		ctx.ptr = nil
	}
}

// Upscale upscales an image
func (ctx *UpscalerContext) Upscale(inputImage SDImage, upscaleFactor uint32) SDImage {
	return *ctx.sd.upscale(ctx.ptr, &inputImage, upscaleFactor)
}

// GetUpscaleFactor gets the upscale factor
func (ctx *UpscalerContext) GetUpscaleFactor() int {
	return int(ctx.sd.getUpscaleFactor(ctx.ptr))
}
