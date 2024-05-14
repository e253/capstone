#ifdef __cplusplus
extern "C" {
#endif

// Loads `OpenCL.dll` or `libopencl.so`
// It will reload the library if it was already loaded
// Returns 0 on success, 1 on failure
bool rocl_init();

// Frees `OpenCL.dll` / `libopencl.so`
// Safe even if `rocl_init` failed
void rocl_deinit();

#ifdef __cplusplus
}
#endif