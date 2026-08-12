#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
static double bench(id<MTLDevice> dev, id<MTLCommandQueue> q, NSUInteger M, NSUInteger K, NSUInteger N) {
    id<MTLBuffer> a = [dev newBufferWithLength:M*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [dev newBufferWithLength:N*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
    memset(a.contents, 1, M*K*2); memset(b.contents, 1, N*K*2);
    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:a descriptor:dA];
    MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:b descriptor:dB];
    MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:c descriptor:dC];
    // C = A * B^T  (weights stored [out][in] = [N][K])
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N
        interiorColumns:K alpha:1.0 beta:0.0];
    double best = 1e30;
    for (int i = 0; i < 8; i++) {
        double t0 = CFAbsoluteTimeGetCurrent();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
        [cb commit]; [cb waitUntilCompleted];
        double dt = CFAbsoluteTimeGetCurrent() - t0;
        if (i >= 3 && dt < best) best = dt;
    }
    return 2.0*M*K*N/best/1e12;
}
struct Shape { NSUInteger M,K,N; const char *name; };
int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        Shape shapes[] = {
            {4096, 4096, 1024, "q_a      (M4096 K4096 N1024)"},
            {4096, 1024, 8192, "q_b      (M4096 K1024 N8192)"},
            {4096, 32768, 1024,"out_low  (M4096 K32768 N1024)"},
            {4096, 1024, 4096, "out_up   (M4096 K1024 N4096)"},
            {2048, 4096, 2048, "exp_mid  (M2048 K4096 N2048)"},
            {96,   4096, 2048, "exp_avg  (M96 K4096 N2048)"},
        };
        for (Shape &s : shapes) {
            double tf = bench(dev, q, s.M, s.K, s.N);
            printf("%-24s %7.2f TFLOP/s-half  (%6.2f ms)\n", s.name, tf, 2.0*s.M*s.K*s.N/(tf*1e12)*1e3);
        }
    }
    return 0;
}
