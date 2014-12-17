#include <stdio.h>
#include <stdlib.h>
// This include pulls in everything you need to develop with OpenCL in OS X.
#include <OpenCL/opencl.h>
// Include the header file generated by Xcode. This header file contains the
// kernel block declaration. // 1
#include "mykernel.cl.h"
// Hard-coded number of values to test, for convenience.
#include <time.h>


int main (int argc, const char * argv[]) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
                         &device_id, &ret_num_devices);
    
    if (device_id == NULL) {
        fprintf(stderr, "failed to get device id!");
        return 1;
    }
    
    printf("%d devices returned \n", ret_num_devices);
  
    // Create sub-device properties: Equally with 4 compute units each:
    cl_device_partition_property props[3];
    props[0] = CL_DEVICE_PARTITION_EQUALLY;  // Equally
    props[1] = 4;                            // 4 compute units per sub-device
    props[2] = 0;                            // End of the property list
    
    cl_device_id subdevice_id[8];
    cl_uint num_entries = 8;
    
    // Create the sub-devices:
    
    cl_int error = clCreateSubDevices(device_id, props, num_entries, subdevice_id, &ret_num_devices);
    
    // Create the context:
    
//    context = clCreateContext(cprops, 1, subdevice_id, NULL, NULL, &err);
    
//    const cl_device_partition_property properties[3] = {
//        CL_DEVICE_PARTITION_BY_COUNTS,
//        1, // Use only one compute unit
//        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END
//    };
//    
//    cl_device_id subdevice_id;
//    cl_int error = clCreateSubDevices(device_id, properties, 1, &subdevice_id, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "failed to create sub device %d!\n", error);
        return 1;
    }
    
    // First, try to obtain a dispatch queue that can send work to the
    // GPU in our system. // 2
    dispatch_queue_t queue =
    gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    // In the event that our system does NOT have an OpenCL-compatible GPU,
    // we can use the OpenCL CPU compute device instead.
    if (queue == NULL) {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    
    int k = 10000; // number of iterations
    float d = 0.85; // damping factor
    
    FILE *data;
    data = fopen("/Users/yixing/Desktop/hollins.dat", "r");
    if (!data) {
        fprintf(stderr, "cannot open datafile\n");
        return 1;
    }
    
    int numNodes, numEdges;
    fscanf(data, "%d %d", &numNodes, &numEdges);
    
    int* numOutLinks = (int*)malloc(sizeof(cl_int) * numNodes);
    int* numInLinks = (int*)malloc(sizeof(cl_int) * numNodes);
    
    int* inlinks = (int*)malloc(sizeof(cl_int) * numEdges);
    int* outlinks = (int*)malloc(sizeof(cl_int) * numEdges);
    int in, out;
    for (int i = 0; i < numEdges; ++i) {
        if(fscanf(data, "%d %d", &in, &out) != EOF) {
            // in and out starts from 1
            // change to let them start from 0
            --in; --out;
            inlinks[i] = (cl_int)in;
            outlinks[i] = (cl_int)out;
            ++numOutLinks[in];
            ++numInLinks[out];
        }
    }
    fclose(data);
    
    int curOffset = 0;
    int* pointers = (int*)malloc(sizeof(cl_int) * numNodes * 2);
    for (int i = 0; i < numNodes; ++i) {
        pointers[2 * i] = curOffset;
        curOffset += numInLinks[i];
        pointers[2 * i + 1] = curOffset;
    }
    
    for (int i = 0; i < numNodes; ++i) {
//        printf("%d %d\n", pointers[2 * i], pointers[2 * i + 1]);
    }
    
    printf("numNodes %d numEdges %d \n", numNodes, numEdges);
    
//  test reading is correct
//    printf("%d\n", numOutLinks[6004]);
//    for (int i = 0; i < numEdges; ++i) {
//        printf("%d %d\n", inlinks[i], outlinks[i]);
//    }
    
    // validate numOutLinks
//    for (int i = 0; i < numNodes; ++i) {
//        printf("%d\n", numOutLinks[i]);
//    }
    
    float* oldpr = (float*)malloc(sizeof(cl_float) * numNodes);
    float* newpr = (float*)malloc(sizeof(cl_float) * numNodes);
    float initPR = 1 / (float)numNodes;
    float constPart = (1 - d) / numNodes;
    printf("const part is %f\n", constPart);
    for (int i = 0; i < numNodes; ++i) {
        oldpr[i] = (cl_float)initPR;
        newpr[i] = constPart;
    }
    
    clock_t t = clock();
    
    void* gcl_oldpr = gcl_malloc(sizeof(cl_float) * numNodes, NULL,
                                 CL_MEM_READ_ONLY);
    void* gcl_newpr = gcl_malloc(sizeof(cl_float) * numNodes, oldpr,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    void* gcl_inlinks = gcl_malloc(sizeof(cl_int) * numEdges, inlinks,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* gcl_outlinks = gcl_malloc(sizeof(cl_int) * numEdges, outlinks,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* gcl_numOutlinks = gcl_malloc(sizeof(cl_int) * numNodes, numOutLinks,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    void* gcl_pointers = gcl_malloc(sizeof(cl_int) * 2 * numNodes, pointers, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    // Dispatch the kernel block using one of the dispatch_ commands and the
    // queue created earlier. // 5
    dispatch_sync(queue, ^{
        // Although we could pass NULL as the workgroup size, which would tell
        // OpenCL to pick the one it thinks is best, we can also ask
        // OpenCL for the suggested size, and pass it ourselves.
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(pagerank_kernel,
                                            CL_KERNEL_WORK_GROUP_SIZE,
                                            sizeof(wgs), &wgs, NULL);
        printf("work group size %d \n", (int)wgs);
        
        // The N-Dimensional Range over which we'd like to execute our
        // kernel. In this case, we're operating on a 1D buffer, so
        // it makes sense that the range is 1D.
        cl_ndrange range1 = { // 6
            1, // The number of dimensions to use.
            {0, 0, 0}, // The offset in each dimension. To specify
            // that all the data is processed, this is 0
            // in the test case. // 7
            {numNodes, 0, 0}, // The global range—this is how many items
            // IN TOTAL in each dimension you want to
            // process.
            {NULL, 0, 0} // The local size of each workgroup. This
            // determines the number of work items per
            // workgroup. It indirectly affects the
            // number of workgroups, since the global
            // size / local size yields the number of
            // workgroups. In this test case, there are
            // NUM_VALUE / wgs workgroups.
        };
        
        cl_ndrange range2 = { // 6
            1, // The number of dimensions to use.
            {0, 0, 0}, // The offset in each dimension. To specify
            // that all the data is processed, this is 0
            // in the test case. // 7
            {numNodes, 0, 0}, // The global range—this is how many items
            // IN TOTAL in each dimension you want to
            // process.
            {NULL, 0, 0} // The local size of each workgroup. This
            // determines the number of work items per
            // workgroup. It indirectly affects the
            // number of workgroups, since the global
            // size / local size yields the number of
            // workgroups. In this test case, there are
            // NUM_VALUE / wgs workgroups.
        };
        
        // Calling the kernel is easy; simply call it like a function,
        // passing the ndrange as the first parameter, followed by the expected
        // kernel parameters. Note that we case the 'void*' here to the
        // expected OpenCL types. Remember, a 'float' in the
        // kernel, is a 'cl_float' from the application's perspective. // 8

        exchange_kernel(&range2, gcl_oldpr, gcl_newpr, gcl_numOutlinks);
        gcl_memcpy(newpr, gcl_newpr, sizeof(cl_float) * numNodes);
        gcl_memcpy(oldpr, gcl_oldpr, sizeof(cl_float) * numNodes);
        for (int i = 0; i < numNodes; ++i) {
//            printf("%f\n", newpr[i]);
        }
        
        for (int i = 0; i < k - 1; ++i) {
            clock_t t1 = clock();
            pagerank_kernel(&range1, gcl_pointers, gcl_inlinks, gcl_oldpr, gcl_newpr);
            t1 = clock() - t1;
            double time_taken = ((double)t1)/CLOCKS_PER_SEC; // in seconds
//            printf("pagerank() took %f seconds to execute \n", time_taken);
            
            clock_t t2 = clock();
            exchange_kernel(&range2, gcl_oldpr, gcl_newpr, gcl_numOutlinks);
            t2 = clock() - t2;
            time_taken = ((double)t2)/CLOCKS_PER_SEC; // in seconds
//            printf("copy took %f seconds to execute \n", time_taken);
        }
        
        // kth iteration
        pagerank_kernel(&range1, gcl_pointers, gcl_inlinks, gcl_oldpr, gcl_newpr);
        gcl_memcpy(newpr, gcl_newpr, sizeof(cl_float) * numNodes);
        
        // Getting data out of the device's memory space is also easy;
        // use gcl_memcpy. In this case, gcl_memcpy takes the output
        // computed by the kernel and copies it over to the
        // application's memory space. // 9
//        gcl_memcpy(test_out, mem_out, sizeof(cl_float) * NUM_VALUES);
    });
    
    for (int i = 0; i < numNodes; ++i) {
        printf("node %d %f\n", i, newpr[i]);
    }
    
    free(numOutLinks);
    free(inlinks);
    free(outlinks);
    free(oldpr);
    free(newpr);
    free(pointers);
    
    // Don't forget to free up the CL device's memory when you're done. // 10
    gcl_free(gcl_newpr);
    gcl_free(gcl_oldpr);
    gcl_free(gcl_inlinks);
    gcl_free(gcl_outlinks);
    gcl_free(gcl_numOutlinks);
    gcl_free(gcl_pointers);
    
    // And the same goes for system memory, as usual.
    // Finally, release your queue just as you would any GCD queue. // 11
    dispatch_release(queue);
    
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("overall time %f\n", time_taken);
}
