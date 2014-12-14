
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
//    newpr[out] = (*d) * oldpr[in] / numOutlinks[in];
    float contribution = (*d) * oldpr[in] / numOutlinks[in];
    if(contribution == 0)
        printf("!!!\n");
    AtomicAdd(&newpr[out], contribution);
//    printf("%f\n", contribution);
}
// 1
// 2
// 3