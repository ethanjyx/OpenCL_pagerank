
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
                   global float* newpr)
{
    size_t i = get_global_id(0);
    int in = inlinks[i];
    int out = outlinks[i];
    AtomicAdd(&newpr[out], oldpr[in]);
}

kernel void exchange(global float* oldpr,
                       global float* newpr,
                       global int* numOutlinks)
{
    size_t i = get_global_id(0);
    if (numOutlinks[i])
        oldpr[i] = 0.85 * newpr[i] / numOutlinks[i];
    else
        oldpr[i] = 0;
    newpr[i] = 0.000025;
}