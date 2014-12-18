
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

kernel void pagerank(global int* pointers,
                   global int* inlinks,
                   global float* oldpr,
                   global float* newpr)
{
    size_t i = get_global_id(0);
    int index = 2 * i;
    int start = pointers[index];
    int end = pointers[index + 1];
//    newpr[i] += start;
    for(int j = start; j < end; ++j) {
        newpr[i] += oldpr[inlinks[j]];
    }
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
    newpr[i] = 1 / 5716808;
}