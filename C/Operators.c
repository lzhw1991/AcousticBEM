/* --------------------------------------------------------------------------- *
 * Copyright (C) 2017 Frank Jargstorff                                         *
 *                                                                             *
 * This file is part of the AcousticBEM library.                               *
 * AcousticBEM is free software: you can redistribute it and/or modify         *
 * it under the terms of the GNU General Public License as published by        *
 * the Free Software Foundation, either version 3 of the License, or           *
 * (at your option) any later version.                                         *
 *                                                                             *
 * AcousticBEM is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of              *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
 * GNU General Public License for more details.                                *
 *                                                                             *
 * You should have received a copy of the GNU General Public License           *
 * along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.        *
 * --------------------------------------------------------------------------- */
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include "gsl/gsl_complex.h"
#include "gsl/gsl_complex_math.h"
#include "gsl/gsl_sf_bessel.h"

#include "Operators.h"

#define MAX_LINE_RULE_SAMPLES 2048

typedef struct {
  float re;
  float im;
} Complex;

float aX_1D[] = {0.980144928249f,     0.898333238707f, 0.762766204958f, 0.591717321248f,
		 0.408282678752f,     0.237233795042f, 0.101666761293f, 1.985507175123E-02f};
float aW_1D[] = {5.061426814519E-02f, 0.111190517227f, 0.156853322939f, 0.181341891689f,
		 0.181341891689f,     0.156853322939f, 0.111190517227f, 5.061426814519E-02f};

float aX_2D[] = {0.333333333333f,     0.797426985353f, 0.101286507323f, 0.101286507323f,
		 0.470142064105f,     0.470142064105f, 0.059715871789f};
float aY_2D[] = {0.333333333333f,     0.101286507323f, 0.797426985353f, 0.101286507323f,
		 0.470142064105f,     0.059715871789f, 0.470142064105f};
float aW_2D[] = {0.225000000000f,     0.125939180544f, 0.125939180544f, 0.125939180544f,
		 0.132394152788f,     0.132394152788f, 0.132394152788f};

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

Float3 add3f(Float3 a, Float3 b) {
  Float3 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

Float2 sub2f(Float2 a, Float2 b) {
  Float2 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

Float3 sub3f(Float3 a, Float3 b) {
  Float3 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

Float2 smul2f(float a, Float2 x) {
  Float2 result;
  result.x = a * x.x;
  result.y = a * x.y;
  return result;
}

Float3 smul3f(float a, Float3 x) {
  Float3 result;
  result.x = a * x.x;
  result.y = a * x.y;
  result.z = a * x.z;
  return result;
}

float dot2f(Float2 a, Float2 b) {
  return a.x * b.x + a.y * b.y;
}

float dot3f(Float3 a, Float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Float3 cross(Float3 a, Float3 b) {
  Float3 result = {a.y * b.z - a.z * b.y,
		   a.z * b.x - a.x * b.z,
		   a.x * b.y - a.y * b.x};
  return result;
}

float norm2f(Float2 a) {
  return sqrtf(dot2f(a, a));
}

float norm3f(Float3 a) {
  return sqrtf(dot3f(a, a));
}

Float2 Normal2D(Float2 a, Float2 b) {
  Float2 vec = sub2f(a, b);
  float len = norm2f(vec);
  Float2 res;
  res.x =  vec.y / len;
  res.y = -vec.x / len;
  
  return res;
}

Float3 Normal3D(Float3 a, Float3 b, Float3 c) {
  Float3 ab = sub3f(b, a);
  Float3 ac = sub3f(c, a);
  Float3 res = cross(ab, ac);
  return smul3f(1.0f/norm3f(res), res);
}

complex float complexQuad2D(complex float(*integrand)(Float2, void*), void * state, IntRule1D intRule,
			    Float2 start, Float2 end) {
  Float2 vec;
  vec = sub2f(end, start);
  complex float sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i)
    sum += intRule.pW[i] * integrand(add2f(smul2f(intRule.pX[i], vec), start), state);

  return norm2f(vec) * sum;
}

complex float complexQuad3D(complex float(*integrand)(Float3, void*), void * state, IntRule2D intRule,
			    Float3 a, Float3 b, Float3 c) {
  Float3 vec_b = sub3f(b, a);
  Float3 vec_c = sub3f(c, a);
  complex float sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    Float3 x = add3f(a, smul3f(intRule.pX[i], vec_b));
    x = add3f(x, smul3f(intRule.pY[i], vec_c));
    sum += intRule.pW[i] * integrand(x, state);
  }
  return 0.5f * norm3f(cross(vec_b, vec_c)) * sum;
}

void semiCircleIntegralRule(int nSections, IntRule1D intRule, IntRule2D * pSemiCircleRule) {
  pSemiCircleRule->nSamples = nSections * intRule.nSamples;
        
  float factor = M_PI / nSections;
  for (int i = 0; i < pSemiCircleRule->nSamples; ++i) {
    float arcAbscissa = (i / intRule.nSamples + intRule.pX[i % intRule.nSamples]) * factor;
    pSemiCircleRule->pX[i] = cosf(arcAbscissa);
    pSemiCircleRule->pY[i] = sinf(arcAbscissa);
    pSemiCircleRule->pW[i] = intRule.pW[i % intRule.nSamples] * factor;
  }
}

complex float complexLineIntegral(complex float(*integrand)(Float2, void*), void * state,
				  IntRule2D intRule) {
  complex float sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    Float2 x = {intRule.pX[i], intRule.pY[i]};
    sum += intRule.pW[i] * integrand(x, state);
  }
  return sum;
}

/* --------------------------------------------------------------------------  */
/*                       2D discrete Helmholtz operators.                      */
/* --------------------------------------------------------------------------- */

typedef struct {
  float k;
  Float2 p;
  Float2 normal_p;
  Float2 normal_q;
} IntL;

complex float intL1_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm2f(sub2f(s->p, x));
  return 0.5f * M_1_PI * logf(R) + 0.25f * I * hankel1(0, s->k * R);
}

complex float intL2_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm2f(sub2f(s->p, x));
  return logf(R);
}

complex float intL3_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm2f(sub2f(s->p, x));
  return hankel1(0, s->k * R);
}

/* Computes elements of the Helmholtz L-Operator for 2D.
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
complex float computeL_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  IntL stat = {k, p};
  IntRule1D intRule = {8, aX_1D, aW_1D};
  
  Float2 ab = sub2f(b, a);
  complex float res;
  if (pOnElement) {
    if (k == 0.0f) {
      float RA = norm2f(sub2f(p, a));
      float RB = norm2f(sub2f(p, b));
      float RAB = norm2f(ab);
      return 0.5f * M_1_PI * (RAB - (RA * logf(RA) + RB * logf(RB)));
    } else {
      return complexQuad2D(intL1_2D, &stat, intRule, a, p) + complexQuad2D(intL1_2D, &stat, intRule, p, b)
	+ computeL_2D(0, p, a, b, true);
    }
  } else {
    if (k == 0.0f) {
      return -0.5f * M_1_PI * complexQuad2D(intL2_2D, &stat, intRule, a, b);
    } else {
      return 0.25f * I * complexQuad2D(intL3_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeL_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeL_2D(k, p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


complex float intM1_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  return dot2f(r, s->normal_q) / dot2f(r, r);
}

complex float intM2_2D(Float2 x, void* state) {
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
complex float computeM_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  Float2 zero = {0.0f, 0.0f};
  IntL stat = {k, p, zero, Normal2D(a, b)};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  complex float res;
  if (pOnElement) {
    return 0.0;
  } else {
    if (k == 0.0f) {
      return -0.5f * M_1_PI * complexQuad2D(intM1_2D, &stat, intRule, a, b);
    } else {
      return 0.25f * I * k * complexQuad2D(intM2_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeM_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeM_2D(k, p, a, b, pOnElement);
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
complex float computeMt_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement) {
  /* The flollowing is a little hacky, as we're not storing the actual normal_p vector in the 
   * normal_q field of the state struct. By doing this we can reuse the two functions for the 
   * M operator's integral evaluation intM1 and intM2.
   */
  Float2 zero = {0.0f, 0.0f};
  IntL stat = {k, p, zero, normal_p};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  complex float res;
  if (pOnElement) {
    return 0.0;
  } else {
    if (k == 0.0f) {
      return -0.5f * M_1_PI * complexQuad2D(intM1_2D, &stat, intRule, a, b);
    } else {
      return -0.25f * I * k * complexQuad2D(intM2_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeMt_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeMt_2D(k, p, normal_p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


complex float intN1_2D(Float2 x, void* state) {
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

complex float intN2_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = sub2f(s->p, x);
  float R2 = dot2f(r, r);
  float drdudrdn = -dot2f(r, s->normal_q) * dot2f(r, s->normal_p) / R2;
  float dpnu = dot2f(s->normal_p, s->normal_q);

  return (dpnu + 2.0f * drdudrdn) / R2;
}

complex float intN3_2D(Float2 x, void* state) {
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
complex float computeN_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement) {
  IntL stat = {k, p, normal_p, Normal2D(a, b)};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  complex float res;
  if (pOnElement) {
    if (k == 0.0f) {
      float RA = norm2f(sub2f(p, a));
      float RB = norm2f(sub2f(p, b));
      float RAB = norm2f(sub2f(b, a));
      return -(1.0f / RA + 1.0f / RB) / (RAB * 2.0 * M_PI) * RAB;
    } else {
      return computeN_2D(0.0f, p, normal_p, a, b, true)
	- 0.5f * k * k * computeL_2D(0.0f, p, a, b, true)
	+ complexQuad2D(intN1_2D, &stat, intRule, a, p) + complexQuad2D(intN1_2D, &stat, intRule, p, b);
    }
  } else {
    if (k == 0.0f) {
      return 0.5 * M_1_PI * complexQuad2D(intN2_2D, &stat, intRule, a, b);
    } else {
      return 0.25f * I * k * complexQuad2D(intN3_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeN_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeN_2D(k, p, normal_p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}

/* --------------------------------------------------------------------------  */
/*         Radially symmetrical discrete Helmholtz operators.                  */
/* --------------------------------------------------------------------------- */

typedef struct {
  float k;
  Float3 p;
  Float2 np;
  Float2 nq;

  float r;
  float z;
  
  IntRule2D semiCircleRule;

  int direction;
} RadIntL;


complex float integrateSemiCircleL_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm3f(sub3f(q, pS->p));

  return cexpf(I * pS->k * R) / R;
}

complex float integrateGeneratorL_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleL_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

complex float integrateSemiCircleL0_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm3f(sub3f(q, pS->p));

  return 1.0f / R;
}

complex float integrateGeneratorL0_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleL0_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

complex float integrateSemiCircleL0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm3f(sub3f(q, pS->p));

  return (cexpf(I * pS->k * R) - 1.0f) / R;
}

complex float integrateGeneratorL0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleL0pOn_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

/* Computes elements of the Helmholtz L-Operator radially symetrical cases.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The complex-valued result of the integration.
 */
complex float computeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = smul2f(0.5, add2f(a, b));
  Float2 ab = sub2f(b, a);
  int nSections = 1 + (int)(q.x * M_PI / norm2f(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  RadIntL state = {k};
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  
  if (pOnElement) {
    assert(8 * 2 * nSections < MAX_LINE_RULE_SAMPLES);
    IntRule2D semiCircleRule = {8 * 2 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
    semiCircleIntegralRule(2 * nSections, intRule, &semiCircleRule);
    state.semiCircleRule = semiCircleRule;

    if (k == 0.0f) {
      return complexQuad2D(integrateGeneratorL0_RAD, &state, intRule, p, a)
	+ complexQuad2D(integrateGeneratorL0_RAD, &state, intRule, p, b);
    } else {
      return computeL_RAD(0.0f, p, a, b, true)
	+ complexQuad2D(integrateGeneratorL0pOn_RAD, &state, intRule, a, b);
    }
  } else {
    assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
    IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
    semiCircleIntegralRule(nSections, intRule, &semiCircleRule);
    state.semiCircleRule = semiCircleRule;

    if (k == 0.0f) {
      return complexQuad2D(integrateGeneratorL0_RAD, &state, intRule, a, b);
    } else {
      return complexQuad2D(integrateGeneratorL_RAD, &state, intRule, a, b);
    }
  }
}

void ComputeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeL_RAD(k, p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}

/* ---------------------------------------------------------------------------
 * Operator M
 */

complex float integrateSemiCircleM_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r = sub3f(q, pS->p);
  float R = norm3f(r);

  return (I * pS->k * R - 1.0f) * cexpf(I * pS->k * R) * dot3f(r, nq) / (R * dot3f(r, r));
}

complex float integrateGeneratorM_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleM_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

complex float integrateSemiCircleMpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r = sub3f(q, pS->p);
  float R = norm3f(r);

  return -dot3f(r, nq) / (R * dot3f(r, r));
}

complex float integrateGeneratorMpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleMpOn_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}


/* Computes elements of the Helmholtz M-Operator radially symetrical cases.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The complex-valued result of the integration.
 */
complex float computeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = smul2f(0.5, add2f(a, b));
  Float2 ab = sub2f(b, a);
  int nSections = 1 + (int)(q.x * M_PI / norm2f(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
  IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
  semiCircleIntegralRule(nSections, intRule, &semiCircleRule);

  RadIntL state = {k};
  state.semiCircleRule = semiCircleRule;
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  state.nq  = Normal2D(a, b);

  if (k == 0.0f) {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, a, b);
    }
  } else {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorM_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorM_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorM_RAD, &state, intRule, a, b);
    }
  }
}


void ComputeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeM_RAD(k, p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}

/* ---------------------------------------------------------------------------
 * Operator Mt
 */

complex float integrateSemiCircleMt_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 r = sub3f(q, pS->p);
  float R = norm3f(r);
  float dotRnP = pS->np.x * r.x + pS->np.y * r.z;
  return -(I * pS->k * R - 1.0f) * cexpf(I * pS->k * R) * dotRnP / (R * dot3f(r, r));
}

complex float integrateGeneratorMt_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleMt_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

complex float integrateSemiCircleMtpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 r = sub3f(q, pS->p);
  float R = norm3f(r);
  float dotRnP = pS->np.x * r.x + pS->np.y * r.z;
  return dotRnP / (R * dot3f(r, r));
}

complex float integrateGeneratorMtpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleMtpOn_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

/* Computes elements of the Helmholtz Mt-Operator radially symetrical cases.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The complex-valued result of the integration.
 */
complex float computeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = smul2f(0.5, add2f(a, b));
  Float2 ab = sub2f(b, a);
  int nSections = 1 + (int)(q.x * M_PI / norm2f(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
  IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
  semiCircleIntegralRule(nSections, intRule, &semiCircleRule);

  RadIntL state = {k};
  state.semiCircleRule = semiCircleRule;
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  state.nq  = Normal2D(a, b);
  state.np  = vec_p;

  if (k == 0.0f) {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, a, b);
    }
  } else {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, a, b);
    }
  }
}


void ComputeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeMt_RAD(k, p, vec_p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}

/* ---------------------------------------------------------------------------
 * Operator N
 */

complex float integrateSemiCircleN_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q  = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r = sub3f(q, pS->p);
  float R = norm3f(r);
  float dotnPnQ = pS->np.x * nq.x + pS->np.y * nq.z;
  float dotRnP  = pS->np.x * r.x + pS->np.y * r.z;
  float dotRnQ  = -dot3f(r, nq);
  float RnPRnQ  = dotRnP * dotRnQ / dot3f(r, r);
  float RnPnQ   = -(dotnPnQ + RnPRnQ) / R;
  complex float ikr = I * pS->k * R;
  complex float fpgr = cexpf(ikr) / dot3f(r, r) * (ikr - 1.0f);
  complex float fpgrr = cexpf(ikr) * (2.0f - 2.0f * ikr - (pS->k * R)*(pS->k * R)) / (dot3f(r, r) * R);
  return fpgr * RnPnQ + fpgrr * RnPRnQ;
}

complex float integrateGeneratorN_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleN_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

complex float integrateSemiCircleNpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q  = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r  = sub3f(q, pS->p);
  float  R  = norm3f(r);
  float dotnPnQ = pS->np.x * nq.x + pS->np.y * nq.z;
  float dotRnP  = pS->np.x * r.x + pS->np.y * r.z;
  float dotRnQ  = -dot3f(r, nq);
  float RnPRnQ  = dotRnP * dotRnQ / dot3f(r, r);
  float RnPnQ   = -(dotnPnQ + RnPRnQ) / R;
  complex float ikr    = I * pS->k * R;
  float         fpg0   = 1.0f / R;
  complex float fpgr   = cexpf(ikr) / dot3f(r, r) * (ikr - 1.0f);
  float         fpgr0  = -1.0f / dot3f(r, r);
  complex float fpgrr  = cexpf(ikr) * (2.0f - 2.0f * ikr - (pS->k * R)*(pS->k * R)) / (dot3f(r, r) * R);
  float         fpgrr0 = 2.0f / (R * dot3f(r, r));
  return (fpgr-fpgr0) * RnPnQ + (fpgrr-fpgrr0) * RnPRnQ + 0.5f * (pS->k*pS->k) * fpg0;
}

complex float integrateGeneratorNpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleNpOn_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}                
		   
complex float integrateSemiCircleN0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q  = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {x.x, x.y, pS->direction};
  nq = smul3f(sqrtf(0.5f), nq);
  Float3 r  = sub3f(q, pS->p);
  float  R  = norm3f(r);
  float dotnPnQ = pS->np.x * nq.x + pS->np.y * nq.z;
  float dotRnP  = pS->np.x * r.x + pS->np.y * r.z;
  float dotRnQ  = -dot3f(r, nq);
  float RnPRnQ  = dotRnP * dotRnQ / dot3f(r, r);
  return (dotnPnQ + 3.0f * RnPRnQ) / (R * dot3f(r,r));
}

complex float integrateGeneratorN0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;
  
  return complexLineIntegral(integrateSemiCircleN0pOn_RAD, pS, pS->semiCircleRule) * pS->r / (2.0f * M_PI);
}

complex float complexConeIntegral(complex float(*integrand)(Float2, void*), void* state, IntRule1D intRule,
				  Float2 start, Float2 end, int nSections) {
  Float2 delta = smul2f(1.0f/nSections, sub2f(end, start));
  complex float sum = 0.0f;
  for (int s = 0; s < nSections; ++s) {
    Float2 segmentStart = add2f(start, smul2f(s, delta));
    Float2 segmentEnd   = add2f(start, smul2f(s+1, delta));
    sum += complexQuad2D(integrand, state, intRule, segmentStart, segmentEnd);
  }
  return sum;
}

/* Computes elements of the Helmholtz N-Operator radially symetrical cases.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The complex-valued result of the integration.
 */
complex float computeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = smul2f(0.5, add2f(a, b));
  Float2 ab = sub2f(b, a);
  int nSections = 1 + (int)(q.x * M_PI / norm2f(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
  IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
  semiCircleIntegralRule(nSections, intRule, &semiCircleRule);

  RadIntL state = {k};
  state.semiCircleRule = semiCircleRule;
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  state.nq  = Normal2D(a, b);
  state.np  = vec_p;

  if (k == 0.0f) {
    if (pOnElement) {
      float lenAB = norm2f(sub2f(b, a));
      /* deal with the cone at the a-side of the generator */
      int direction = -1;
      if (a.y >= b.y) direction = 1;
      Float2 tip_a = {0.0f, a.y + direction * a.x};
      state.direction = direction;
      int nSections = (int)(a.x * sqrtf(2.0f) / lenAB) + 1;
      complex float coneValA = complexConeIntegral(integrateGeneratorN0pOn_RAD, &state, intRule, a, tip_a, nSections);

      /* deal with the cone at the b-side of the generator */
      Float2 tip_b = {0.0, b.y - direction * b.x};
      state.direction = -direction;
      nSections = (int)(b.x * sqrtf(2.0f) / lenAB) + 1;
      complex float coneValB = complexConeIntegral(integrateGeneratorN0pOn_RAD, &state, intRule, b, tip_b, nSections);
      
      return -(coneValA + coneValB);
    } else {
      return 0.0f;
    }
  } else {
    if (pOnElement) {
      return computeN_RAD(0.0f, p, vec_p, a, b, true) - 0.5f * (k*k) * computeL_RAD(0.0f, p, a, b, true)
	+ complexQuad2D(integrateGeneratorNpOn_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorNpOn_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorN_RAD, &state, intRule, a, b);
    }
  }
}

void ComputeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  complex float z = computeN_RAD(k, p, vec_p, a, b, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


/* --------------------------------------------------------------------------  */
/*                  3D discrete Helmholtz operators.                           */
/* --------------------------------------------------------------------------- */

typedef struct {
  float k;
  Float3 p;
  Float3 normal_p;
  Float3 normal_q;
} IntL3D;

complex float intL1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  float R = norm3f(sub3f(s->p, x));
  return cexpf(I * s->k * R) / R;
}

complex float intL2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  float R = norm3f(sub3f(s->p, x));
  return 1.0 / R;
}

complex float intL3_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  float R = norm3f(sub3f(s->p, x));
  return (cexpf(I * s->k * R) - 1.0) / R;
}

 /* Computes elements of the Helmholtz L-Operator for 3D.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   ax, ay, az - the first vertex of ccw triangle.
 *   bx, by, bz - the second vertex of ccw triangle.
 *   cx, cy, cz - the third vertex of ccw t riangle.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};
  
  if (pOnElement) {
    if (k == 0.0f) {
      Float3 ab = sub3f(b, a);
      Float3 ac = sub3f(c, a);
      Float3 bc = sub3f(c, b);

      float aopp[3] = {norm3f(ab), norm3f(bc), norm3f(ac)};

      Float3 ap = sub3f(p, a);
      Float3 bp = sub3f(p, b);
      Float3 cp = sub3f(p, c);

      float ar0[3] = {norm3f(ap), norm3f(bp), norm3f(cp)};
      float ara[3] = {ar0[1], ar0[2], ar0[0]};
      
      float result = 0.0f;
      for (int i = 0; i < 3; ++i) {
	float r0 = ar0[i];
	float ra = ara[i];
	float opp = aopp[i];
	if (r0 < ra) {
	  float temp = r0;
	  r0 = ra;
	  ra = temp;
	}
	float A = acosf((ra*ra + r0*r0 - opp*opp) / (2.0f * ra * r0));
	float B = atanf(ra * sinf(A) / (r0 - ra * cosf(A)));
	result += (r0 * sinf(B) * (logf(tanf(0.5f * (A + B))) - logf(tanf(0.5 * B))));
      }
      return result / (4.0f * M_PI);
    } else {
      complex float L0 = computeL_3D(0.0, p, a, b, c, true);
      complex float Lk = complexQuad3D(intL3_3D, &stat, intRule, a, b, p)
	+ complexQuad3D(intL3_3D, &stat, intRule, b, c, p)
	+ complexQuad3D(intL3_3D, &stat, intRule, c, a, p);
      return L0 + Lk / (4.0f * M_PI);

    }
  } else {
    if (k == 0.0f) {
      return complexQuad3D(intL2_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    } else {
      return complexQuad3D(intL1_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    }
  }
}
        
void ComputeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  complex float z = computeL_3D(k, p, a, b, c, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}

complex float intM1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = sub3f(s->p, x);
  float R = norm3f(r);
  float kr = s->k * R;
  complex float ikr = I * kr;
  float rnq = -dot3f(r, s->normal_q) / R;
  return rnq * (ikr - 1.0f) * cexpf(ikr) / dot3f(r, r);
}

complex float intM2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = sub3f(s->p, x);
  float R = norm3f(r);
  float rnq = -dot3f(r, s->normal_q) / R;
  return -1.0f / dot3f(r, r) * rnq;
}

 /* Computes elements of the Helmholtz M-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   a, b, c - the three vertices forming the triangle being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  stat.normal_q = Normal3D(a, b, c);
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};
  
  if (pOnElement) {
    return 0.0f;
  } else {
    if (k == 0.0f) {
      return complexQuad3D(intM2_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    } else {
      return complexQuad3D(intM1_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    }
  }
}

void ComputeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  complex float z = computeM_3D(k, p, a, b, c, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


complex float intMt1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = sub3f(s->p, x);
  float R = norm3f(r);
  float kr = s->k * R;
  complex float ikr = I * kr;
  float rnp = dot3f(r, s->normal_p) / R;
  return rnp * (ikr - 1.0f) * cexpf(ikr) / dot3f(r, r);
}

complex float intMt2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = sub3f(s->p, x);
  float R = norm3f(r);
  float rnp = dot3f(r, s->normal_p) / R;
  return -1.0f / dot3f(r, r) * rnp;
}

 /* Computes elements of the Helmholtz Mt-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   vec_p - the surface normal in p.
 *   a, b, c - the three vertices forming the triangle being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  stat.normal_p = vec_p;
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};
  
  if (pOnElement) {
    return 0.0f;
  } else {
    if (k == 0.0f) {
      return complexQuad3D(intMt2_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    } else {
      return complexQuad3D(intMt1_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    }
  }
}

void ComputeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  complex float z = computeMt_3D(k, p, vec_p, a, b, c, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}


complex float intN1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = sub3f(s->p, x);
  float R  = norm3f(r);
  float kr = s->k * R;
  complex float ikr = I * kr;

  float rnq    = -dot3f(r, s->normal_q) / R;
  float rnp    =  dot3f(r, s->normal_p) / R;
  float dnpnq  =  dot3f(s->normal_p, s->normal_q);
  float rnprnq = rnp * rnq;
  float rnpnq  = -(dnpnq + rnprnq) / R;

  complex float fpgr  = (ikr - 1.0f) * cexpf(ikr) / dot3f(r, r);
  complex float fpgrr = cexpf(ikr) * (2.0f - 2.0f*ikr - kr*kr) / (R * dot3f(r, r));
  return fpgr * rnpnq + fpgrr * rnprnq;
}

complex float intN2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = sub3f(s->p, x);
  float R  = norm3f(r);
  float kr = s->k * R;
  complex float ikr = I * kr;

  float rnq    = -dot3f(r, s->normal_q) / R;
  float rnp    =  dot3f(r, s->normal_p) / R;
  float dnpnq  =  dot3f(s->normal_p, s->normal_q);
  float rnprnq = rnp * rnq;
  float rnpnq  = -(dnpnq + rnprnq) / R;


          float fpg   = 1.0f / R;
  complex float fpgr  = ((ikr - 1.0f) * cexpf(ikr) + 1.0f) / dot3f(r, r);
  complex float fpgrr = (cexpf(ikr) * (2.0f - 2.0f*ikr - kr*kr) - 2.0f) / (R * dot3f(r, r));
  return fpgr * rnpnq + fpgrr * rnprnq + (0.5f*s->k*s->k) * fpg;
}

/* Computes elements of the Helmholtz N-Operator.
 * 
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   vec_p - the surface normal in point p.
 *   a, b, c - the vertices of the trinagle being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 * 
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the 
 *     second being the Imaginary component of that complex value.
 */
complex float computeN_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  stat.normal_p = vec_p;
  stat.normal_q = Normal3D(a, b, c);
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};

  if (pOnElement) {
    if (k == 0.0f) {
      Float3 ab = sub3f(b, a);
      Float3 ac = sub3f(c, a);
      Float3 bc = sub3f(c, b);

      float aopp[3] = {norm3f(ab), norm3f(bc), norm3f(ac)};

      Float3 ap = sub3f(p, a);
      Float3 bp = sub3f(p, b);
      Float3 cp = sub3f(p, c);

      float ar0[3] = {norm3f(ap), norm3f(bp), norm3f(cp)};
      float ara[3] = {ar0[1], ar0[2], ar0[0]};
      
      float result = 0.0f;
      for (int i = 0; i < 3; ++i) {
	float r0 = ar0[i];
	float ra = ara[i];
	float opp = aopp[i];
	if (r0 < ra) {
	  float temp = r0;
	  r0 = ra;
	  ra = temp;
	}
	float A = acosf((ra*ra + r0*r0 - opp*opp) / (2.0f * ra * r0));
	float B = atanf(ra * sinf(A) / (r0 - ra * cosf(A)));
	result += (cosf(A + B) - cosf(B)) / (r0 * sinf(B));
      }   
      return result / (4.0f * M_PI);
    } else {
      complex float N0 = computeN_3D(0.0f, p, vec_p, a, b, c, true);
      complex float L0 = computeL_3D(0.0f, p, a, b, c, true);
      complex float Nk = complexQuad3D(intN2_3D, &stat, intRule, a, b, p)
	+ complexQuad3D(intN2_3D, &stat, intRule, b, c, p)
	+ complexQuad3D(intN2_3D, &stat, intRule, c, a, p);
	return N0 - (0.5f*k*k) * L0 + Nk / (4.0f * M_PI);
    }
  } else {
    if (k == 0.0f) {
      return 0.0f;
    } else {
      return complexQuad3D(intN1_3D, &stat, intRule, a, b, c) / (4.0f * M_PI);
    }
  }
}

void ComputeN_3D(float k, Float3 p, Float3 normal_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  complex float z = computeN_3D(k, p, normal_p, a, b, c, pOnElement);
  pResult->re = crealf(z);
  pResult->im = cimagf(z);
}
