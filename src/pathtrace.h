#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
//void pathtrace(uchar4 *pbo, int frame, int iteration);

//HW04 
void pathtrace(uchar4* pbo, int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void denoise(float c_phi, float n_phi, float p_phi, int filterSize);
