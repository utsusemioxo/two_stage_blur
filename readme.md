# my opencl wrapper

A complete sequence for executing an OpenCL program is:

1. Query for avaliable OpenCL platforms and devices
2. Create a context for one or more OpenCL devices in a platform
3. Create and build programs for OpenCL devices in a platform
4. Select kernels to execute from the programs
5. Create memory objects for kernels to operate on
6. Create command queues to execute commands on an OpenCL device
7. Enqueue data transfer commands into the memory objects, if needed
8. Enqueue kernels into the command queue for execution
9. Enqueue commands to transfer data back to the host, if needed