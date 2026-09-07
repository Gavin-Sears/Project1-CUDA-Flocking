#pragma once

#define CUDA(call) do {                             \
    cudaError_t e = (call);                         \
    if (e == cudaSuccess) break;                    \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",       \
            __LINE__, cudaGetErrorString(e), e);    \
    exit(1);                                        \
} while (0)

namespace Boids {
    void initSimulation(int N);
    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();
}
