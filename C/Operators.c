#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>


#include "gsl/gsl_complex.h"
#include "gsl/gsl_complex_math.h"
#include "gsl/gsl_sf_bessel.h"


typedef struct {
  float re;
  float im;
} Complex;

typedef struct {
  float x;
  float y;
} Float2;

complex float hankel1(int order, float x) {
  return jnf(order, x) + ynf(order, x) * I;
}

void Hankel1(int order, float x, Complex * pz) {
  complex float z = hankel1(order, x);
  pz->re = crealf(z);
  pz->im = cimagf(z);
}

Float2 add2f(Float2 a, Float2 b) {
  Float2 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  return result;
}

Float2 sub2f(Float2 a, Float2 b) {
  Float2 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

Float2 smul2f(float a, Float2 x) {
  Float2 result;
  result.x = a * x.x;
  result.y = a * x.y;
  return result;
}

float dot2f(Float2 a, Float2 b) {
  return a.x * b.x + a.y * b.y;
}

float norm2f(Float2 a) {
  return sqrtf(dot2f(a, a));
}

Float2 Normal2D(Float2 a, Float2 b) {
  Float2 vec = sub2f(a, b);
  float len = norm2f(vec);
  Float2 res;
  res.x =  vec.y / len;
  res.y = -vec.x / len;
  
  return res;
}

complex float complexQuad(complex float(*integrand)(Float2, void*), void * state, Float2 start, Float2 end) {
  float aX[8] = {0.980144928249,     0.898333238707, 0.762766204958, 0.591717321248,
		 0.408282678752,     0.237233795042, 0.101666761293, 1.985507175123E-02};
  float aW[8] = {5.061426814519E-02, 0.111190517227, 0.156853322939, 0.181341891689,
		 0.181341891689,     0.156853322939, 0.111190517227, 5.061426814519E-02};
  Float2 vec;
  vec = sub2f(end, start);
  complex float sum = 0.0;
  for (int i = 0; i < 8; ++i)
    sum += aW[i] * integrand(add2f(smul2f(aX[i], vec), start), state);

  return norm2f(vec) * sum;
}

typedef struct {
  float k;
  Float2 p;
  Float2 normal_p;
  Float2 normal_q;
} IntL;

complex float intL1(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm2f(sub2f(s->p, x));
  return 0.5f * M_1_PI * logf(R) + 0.25f * I * hankel1(0, s->k * R);
}

complex float intL2(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm2f(sub2f(s->p, x));
  return logf(R);
}

complex float intL3(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm2f(sub2f(s->p, x));
  return hankel1(0, s->k * R);
}

/* Computes elements of the Helmholtz L-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeL(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  Float2 ab = sub2f(b, a);
  IntL stat = {k, p};
  complex float res;
  if (pOnElement) {
    if (k == 0.0f) {
      float RA = norm2f(sub2f(p, a));
      float RB = norm2f(sub2f(p, b));
      float RAB = norm2f(ab);
      return 0.5f * M_1_PI * (RAB - (RA * logf(RA) + RB * logf(RB)));
    } else {
      return complexQuad(intL1, &stat, a, p) + complexQuad(intL1, &stat, p, b)
	+ computeL(0, p, a, b, true);
    }
  } else {
    if (k == 0.0f) {
      return -0.5f * M_1_PI * complexQuad(intL2, &stat, a, b);
    } else {
      return 0.25f * I * complexQuad(intL3, &stat, a, b);
    }
  }
}

void ComputeL(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeL(k, p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


complex float intM1(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  return dot2f(r, s->normal_q) / dot2f(r, r);
}

complex float intM2(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  float R = norm2f(r);
  return hankel1(1, s->k * R) * dot2f(r, s->normal_q) / R;
}

/* Computes elements of the Helmholtz M-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeM(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  Float2 zero = {0.0f, 0.0f};
  IntL stat = {k, p, zero, Normal2D(a, b)};
  complex float res;
  if (pOnElement) {
    return 0.0;
  } else {
    if (k == 0.0f) {
      return -0.5f * M_1_PI * complexQuad(intM1, &stat, a, b);
    } else {
      return 0.25f * I * k * complexQuad(intM2, &stat, a, b);
    }
  }
}

void ComputeM(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeM(k, p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}

/* Computes elements of the Helmholtz Mt-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeMt(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement) {
  /* The flollowing is a little hacky, as we're not storing the actual normal_p vector in the 
   * normal_q field of the state struct. By doing this we can reuse the two functions for the 
   * M operator's integral evaluation intM1 and intM2.
   */
  Float2 zero = {0.0f, 0.0f};
  IntL stat = {k, p, zero, normal_p};
  complex float res;
  if (pOnElement) {
    return 0.0;
  } else {
    if (k == 0.0f) {
      return -0.5f * M_1_PI * complexQuad(intM1, &stat, a, b);
    } else {
      return -0.25f * I * k * complexQuad(intM2, &stat, a, b);
    }
  }
}

void ComputeMt(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeMt(k, p, normal_p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


complex float intN1(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  float R2 = dot2f(r, r);
  float R = sqrtf(R2);
  float drdudrdn = -dot2f(r, s->normal_q) * dot2f(r, s->normal_p) / R2;
  float dpnu = dot2f(s->normal_p, s->normal_q);
  complex float c1 = 0.25f * I * s->k / R * hankel1(1, s->k * R) - 0.5f * M_1_PI / R2;
  complex float c2 = 0.50f * I * s->k / R * hankel1(1, s->k * R)
    - 0.25f * I * s->k * s->k * hankel1(0, s->k * R) - M_1_PI / R2;
  float c3 = -0.25f * s->k * s->k * logf(R) * M_1_PI;
  
  return c1 * dpnu + c2 * drdudrdn + c3;
}

complex float intN2(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  float R2 = dot2f(r, r);
  float drdudrdn = -dot2f(r, s->normal_q) * dot2f(r, s->normal_p) / R2;
  float dpnu = dot2f(s->normal_p, s->normal_q);

  return (dpnu + 2.0f * drdudrdn) / R2;
}

complex float intN3(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  float R2 = dot2f(r, r);
  float R = sqrtf(R2);
  float drdudrdn = -dot2f(r, s->normal_q) * dot2f(r, s->normal_p) / R2;
  float dpnu = dot2f(s->normal_p, s->normal_q);

  return hankel1(1, s->k * R) / R * (dpnu + 2.0 * drdudrdn)
    - s->k * hankel1(0, s->k * R) * drdudrdn;
}


/* Computes elements of the Helmholtz N-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeN(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement) {
  IntL stat = {k, p, normal_p, Normal2D(a, b)};
  complex float res;
  if (pOnElement) {
    if (k == 0.0f) {
      float RA = norm2f(sub2f(p, a));
      float RB = norm2f(sub2f(p, b));
      float RAB = norm2f(sub2f(b, a));
      return -(1.0f / RA + 1.0f / RB) / (RAB * 2.0 * M_PI) * RAB;
    } else {
      return computeN(0.0f, p, normal_p, a, b, true)
	- 0.5f * k * k * computeL(0.0f, p, a, b, true)
	+ complexQuad(intN1, &stat, a, p) + complexQuad(intN1, &stat, p, b);
    }
  } else {
    if (k == 0.0f) {
      return 0.5 * M_1_PI * complexQuad(intN2, &stat, a, b);
    } else {
      return 0.25f * I * k * complexQuad(intN3, &stat, a, b);
    }
  }
}

void ComputeN(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeN(k, p, normal_p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}
