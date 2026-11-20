// LD_PRELOAD shim: report "no peer access" between GPUs so CUDA libraries (PyTorch, NCCL, ...)
// never use direct PCIe peer-to-peer copies, which silently return garbage on machines where the
// IOMMU/ACS blocks P2P. Cross-device copies then go through cudaMemcpy, which the driver stages
// through host memory.
typedef int cudaError_t;
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    (void)device; (void)peerDevice;
    *canAccessPeer = 0;
    return 0;
}
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void)peerDevice; (void)flags;
    return 0;
}
