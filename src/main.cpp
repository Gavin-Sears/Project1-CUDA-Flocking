/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 5650
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"
#include "kernel.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1
// Runtime (not compile-time) so runBenchmark() can sweep through all three
// implementations in one process without a rebuild.
bool UNIFORM_GRID = true;
bool COHERENT_GRID = true;

// Set to true to run the automated performance sweep in runBenchmark()
// instead of the normal interactive loop. Works with VISUALIZE either way.
// With VISUALIZE 1, runBenchmark() renders and presents each frame (see
// renderBenchmarkFrame()), so the results reflect real draw and copy
// overhead. With VISUALIZE 0 it stays fully headless.
const bool RUN_BENCHMARK = false;

// LOOK-1.2 - change this to adjust particle count in the simulation
const int N_FOR_VIS = 5000;
const float DT = 0.2f;

// Boid counts swept for every implementation in runBenchmark(). Declared
// here, before initVAO(), so initVAO() can size the VBOs and IBO to cover
// the largest one (see MAX_VIS_BOIDS below). Block size and grid cell width
// aren't swept here since they're compile-time constants in kernel.cu.
// Change one by hand, rebuild, and rerun the benchmark to get a data point
// for those graphs.
const int BENCHMARK_BOID_COUNTS[] = { 500, 1000, 5000, 10000, 50000, 100000, 500000 };

// initVAO() allocates the boid VBOs and IBO once, at startup, sized for
// however many boids will ever need to be drawn in this run. runBenchmark()
// never resizes them later when it changes boid count via initSimulation().
// Must cover both the normal interactive count and every benchmark sweep
// value. Adds one to safely cover mainLoop()'s `N_FOR_VIS + 1` glDrawElements
// call.
int computeMaxVisBoids() {
  int maxN = N_FOR_VIS;
  for (int n : BENCHMARK_BOID_COUNTS) {
    if (n > maxN) {
      maxN = n;
    }
  }
  return maxN + 1;
}
const int MAX_VIS_BOIDS = computeMaxVisBoids();

// How many boids to actually draw right now. Equals N_FOR_VIS for the whole
// run in normal interactive mode. runBenchmark() updates this whenever it
// changes boid count, so a with-visualization sweep draws the right number
// of points at each step instead of whatever N_FOR_VIS happened to be.
int currentVisBoidCount = N_FOR_VIS;

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "5650 CUDA Intro: Boids";

  if (init(argc, argv)) {
    if (RUN_BENCHMARK) {
      runBenchmark();
    } else {
      mainLoop();
    }
    Boids::endSimulation();
    return 0;
  } else {
    return 1;
  }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO();

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize N-body simulation
  Boids::initSimulation(N_FOR_VIS);

  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

  return true;
}

void initVAO() {

  // Sized to MAX_VIS_BOIDS, not N_FOR_VIS, so these buffers are big enough
  // for every boid count runBenchmark() will ever use, not just the initial
  // interactive one. See the comment on MAX_VIS_BOIDS above.
  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (MAX_VIS_BOIDS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[MAX_VIS_BOIDS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < MAX_VIS_BOIDS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (MAX_VIS_BOIDS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (MAX_VIS_BOIDS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (MAX_VIS_BOIDS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
      glUniform3fv(location, 1, &cameraPosition[0]);
    }
  }

  //====================================
  // Main loop
  //====================================
  void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

    // execute the kernel
    if (UNIFORM_GRID && COHERENT_GRID) {
      Boids::stepSimulationCoherentGrid(DT);
    } else if (UNIFORM_GRID) {
      Boids::stepSimulationScatteredGrid(DT);
    } else {
      Boids::stepSimulationNaive(DT);
    }

    #if VISUALIZE
    Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    #endif
    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
  }

  void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    Boids::unitTest(); // LOOK-1.2 We run some basic example code to make sure
                       // your CUDA development setup is ready to go.

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      frame++;
      double time = glfwGetTime();

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;
      }

      runCUDA();

      std::ostringstream ss;
      ss << "[";
      ss.precision(1);
      ss << std::fixed << fps;
      ss << " fps] " << deviceName;
      glfwSetWindowTitle(window, ss.str().c_str());

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      #if VISUALIZE
      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
      glPointSize(1.0f);

      glUseProgram(0);
      glBindVertexArray(0);

      glfwSwapBuffers(window);
      #endif
    }
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  //====================================
  // Automated performance benchmarking
  //====================================

  struct BenchmarkResult {
    std::string implementation;
    int numBoids;
    double avgFps;
  };

  const double BENCHMARK_WARMUP_SECONDS = 5.0;
  const int BENCHMARK_NUM_SAMPLES = 5;

  // Draws and presents one frame for runBenchmark(), mirroring what
  // mainLoop() does each frame (clear/draw/swap). A separate function from
  // mainLoop() rather than a shared helper, so the already-working
  // interactive path stays untouched. Entirely a no-op when VISUALIZE is 0,
  // same as mainLoop().
  void renderBenchmarkFrame() {
    #if VISUALIZE
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program[PROG_BOID]);
    glBindVertexArray(boidVAO);
    glPointSize((GLfloat)pointSize);
    glDrawElements(GL_POINTS, currentVisBoidCount, GL_UNSIGNED_INT, 0);
    glPointSize(1.0f);
    glUseProgram(0);
    glBindVertexArray(0);
    glfwSwapBuffers(window);
    #endif
  }

  // Runs the simulation for `seconds` of real time without recording
  // anything, so the flock can settle after a fresh initSimulation() before
  // measurement starts.
  void benchmarkWarmup(double seconds) {
    double start = glfwGetTime();
    while (glfwGetTime() - start < seconds && !glfwWindowShouldClose(window)) {
      glfwPollEvents();
      runCUDA();
      renderBenchmarkFrame();
    }
  }

  // Runs the simulation for exactly one second of real time and returns the
  // resulting fps, using the same frame-count / elapsed-time approach as
  // mainLoop()'s title bar counter.
  double benchmarkSampleOneSecond() {
    int frame = 0;
    double timebase = glfwGetTime();
    double time = timebase;
    while (time - timebase < 1.0 && !glfwWindowShouldClose(window)) {
      glfwPollEvents();
      runCUDA();
      renderBenchmarkFrame();
      frame++;
      time = glfwGetTime();
    }
    return frame / (time - timebase);
  }

  void runBenchmark() {
    struct Implementation { const char *name; bool uniformGrid; bool coherentGrid; };
    const Implementation implementations[] = {
      { "naive",     false, false },
      { "scattered", true,  false },
      { "coherent",  true,  true  },
    };

    std::vector<BenchmarkResult> results;

    for (const Implementation &impl : implementations) {
      UNIFORM_GRID = impl.uniformGrid;
      COHERENT_GRID = impl.coherentGrid;

      for (int n : BENCHMARK_BOID_COUNTS) {
        if (glfwWindowShouldClose(window)) {
          break;
        }

        std::cout << "Benchmarking " << impl.name << ", N = " << n << " ..." << std::endl;

        // Reset the CUDA-side buffers to the new boid count. Safe to call
        // endSimulation() here even on the first iteration. init() already
        // ran initSimulation(N_FOR_VIS) once before runBenchmark() started.
        Boids::endSimulation();
        Boids::initSimulation(n);
        currentVisBoidCount = n;

        benchmarkWarmup(BENCHMARK_WARMUP_SECONDS);

        double fpsSum = 0.0;
        for (int i = 0; i < BENCHMARK_NUM_SAMPLES; i++) {
          fpsSum += benchmarkSampleOneSecond();
        }
        double avgFps = fpsSum / BENCHMARK_NUM_SAMPLES;

        results.push_back({ impl.name, n, avgFps });
        std::cout << "  -> " << avgFps << " fps" << std::endl;
      }
    }

    std::ostringstream out;
    out << "implementation,numBoids,avgFps" << std::endl;
    for (const BenchmarkResult &r : results) {
      out << r.implementation << "," << r.numBoids << "," << r.avgFps << std::endl;
    }

    std::cout << std::endl << out.str();

    std::ofstream file("benchmark_results.txt");
    if (file.is_open()) {
      file << out.str();
      file.close();
      std::cout << "Results written to benchmark_results.txt" << std::endl;
    } else {
      std::cout << "Warning: could not open benchmark_results.txt for writing" << std::endl;
    }
  }

  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      // compute new camera parameters
      phi += (xpos - lastX) / width;
      theta -= (ypos - lastY) / height;
      theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
      updateCamera();
    }
    else if (rightMousePressed) {
      zoom += (ypos - lastY) / height;
      zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
      updateCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
  }
