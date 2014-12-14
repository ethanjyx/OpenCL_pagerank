
// atomic add float
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

kernel void square(
                   global int* inlinks,
                   global int* outlinks,
                   global int* numOutlinks,
                   global float* oldpr,
                   global float* newpr,
                   global float* d)
{
    size_t i = get_global_id(0);
    int in = inlinks[i];
    int out = outlinks[i];
    float contribution = (*d) * oldpr[in] / numOutlinks[in] + newpr[out];
    AtomicAdd(&newpr[out], contribution);
}
// 1
// 2
// 3