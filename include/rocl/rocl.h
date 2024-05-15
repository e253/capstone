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

// Returns a string representation of the OpenCL error code
// Thanks Selmar on StackOverflow: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clErrorToString(cl_int err);

#ifdef __cplusplus
}
#endif