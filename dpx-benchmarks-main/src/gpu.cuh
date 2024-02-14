#ifndef GPU_CUH
#define GPU_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <string>
#include "utils.cuh"
#include "logger.cuh"

namespace dpxbench { namespace gpu
{
    class GPUs {
        int device_count;
        int device; // Current selected device
        cudaDeviceProp curr_dev_prop;

        public:
            GPUs () : device_count(0), device(0) {
                cudaSetDevice(0);
                cudaGetDeviceCount(&this->device_count);
                CHECK_LAST_CUDA_ERROR();
                cudaGetDeviceProperties(&this->curr_dev_prop, 0);
            };

            void set_device (const unsigned int dev) {
                if (dev >= this->device_count) {
                    LOG_ERROR("Device %d does not exist", dev);
                    return;
                }
                cudaSetDevice(dev);
                CHECK_LAST_CUDA_ERROR();
                this->device = dev;
                cudaGetDeviceProperties(&this->curr_dev_prop, dev);
            };

            void pprint () {

                // List all available GPUs marking the selected one
                printf("-------- AVAILABLE GPUs --------\n");
                for (int i = 0; i < this->device_count; i++) {
                    cudaDeviceProp dev_prop;
                    cudaGetDeviceProperties(&dev_prop, i);
                    CHECK_LAST_CUDA_ERROR();
                    if (i == this->device) {
                        printf("[*] ");
                    } else {
                        printf("[ ] ");
                    }
                    printf("%d: %s\n", i, dev_prop.name);
                }

                printf("\n");

                printf("-------- COMPILATION INFO --------\n");
                printf("Compiled with CUDA version: %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
                printf("\n");

                printf("-------- GPU --------\n");
                printf("Device %d: %s\n", this->device, this->get_name().c_str());
                printf("Compute capability: %d.%d\n", this->get_major(), this->get_minor());
                printf("Driver version: %d.%d\n", this->get_driver_version() / 1000, this->get_driver_version() % 1000);

                printf("-------- COMPUTE --------\n");
                printf("GPU clock rate: %.2f MHz\n", (double)this->get_clock_rate() / 1000);
                printf("Number of SMs: %d\n", this->get_multi_processor_count());
                printf("Maximum number of blocks per multiprocessor: %d\n", this->get_max_blocks_per_multiprocessor());
                printf("Maximum number of threads per block: %d (%d warps)\n", this->get_max_threads_per_block(), this->get_max_threads_per_block() / 32);
                printf("Maximum number of threads per multiprocessor: %d (%d warps)\n", this->get_max_threads_per_multiprocessor(), this->get_max_threads_per_multiprocessor() / 32);


                printf("-------- MEMORY --------\n");
                printf("Total global memory: %.2f GiB\n", (double)this->get_total_global_memory() / 1024 / 1024 / 1024);
                printf("Peak theoretical global memory bandwidth: %.2f GB/s\n", (double)this->get_peak_memory_gbps());
                printf("L2 cache size: %.2f KiB\n", (double)this->get_l2_cache_size() / 1024);
                printf("Shared memory available per block: %.2f KiB\n", (double)this->get_shared_memory_per_block() / 1024);
                printf("Shared memory per multiprocessor: %.2f KiB\n", (double)this->get_sh_mem_per_multiprocessor() / 1024);
                printf("Registers per multiprocessor: %d\n", this->get_regs_per_multiprocessor());
                printf("Registers per block: %d\n", this->get_registers_per_block());
                printf("-------- --------\n");

            }

            std::string get_name () {
                return std::string(this->curr_dev_prop.name);
            };

            // Compute capabilities
            int get_driver_version () {
                int driver_version;
                cudaDriverGetVersion(&driver_version);
                CHECK_LAST_CUDA_ERROR();
                return driver_version;
            };
            int get_major () {
                return this->curr_dev_prop.major;
            };
            int get_minor () {
                return this->curr_dev_prop.minor;
            };


            // Memory
            size_t get_total_global_memory () {
                return this->curr_dev_prop.totalGlobalMem;
            };
            double get_peak_memory_gbps () {
                // In GB/s
                // TODO: memoryClockRate is deprecated
                return (double)(2.0 * (double)this->curr_dev_prop.memoryClockRate * 1000.0 * (double)this->curr_dev_prop.memoryBusWidth / 8.0) / 1.0e9;
            };
            size_t get_shared_memory_per_block () {
                return this->curr_dev_prop.sharedMemPerBlock;
            };
            int get_registers_per_block () {
                return this->curr_dev_prop.regsPerBlock;
            };
            int get_l2_cache_size () {
                // In bytes
                return this->curr_dev_prop.l2CacheSize;
            };

            // Architecture / compute
            int get_multi_processor_count () {
                return this->curr_dev_prop.multiProcessorCount;
            };
            int get_sh_mem_per_multiprocessor () {
                return this->curr_dev_prop.sharedMemPerMultiprocessor;
            };
            int get_regs_per_multiprocessor () {
                return this->curr_dev_prop.regsPerMultiprocessor;
            };
            int get_clock_rate () {
                // TODO: clockRate is deprecated
                return this->curr_dev_prop.clockRate;
            };


            // Blocks / threads
            int get_max_threads_per_block () {
                return this->curr_dev_prop.maxThreadsPerBlock;
            };
            int get_max_threads_per_multiprocessor () {
                return this->curr_dev_prop.maxThreadsPerMultiProcessor;
            };
            int get_max_blocks_per_multiprocessor () {
                return this->curr_dev_prop.maxBlocksPerMultiProcessor;
            };
    };

}} // namespace

#endif // GPU_CUH