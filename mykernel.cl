// Simple OpenCL kernel that squares an input array.
// This code is stored in a file called mykernel.cl.
// You can name your kernel file as you would name any other
// file.  Use .cl as the file extension for all kernel
// source files.
// Kernel block.
kernel void pagerank(
                   global int* oldpr,
                   global int* newpr,
                   global int* inlinks,
                   global int* outlinks,
                   )
{
    size_t i = get_global_id(0);
    output[i] = input[i] * input[i];
}
// 1
// 2
// 3