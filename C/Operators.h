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
#pragma once

#include <complex.h>

/* Struct representing points in 2-dimensional Euclidian space. */
typedef struct {
  float x;
  float y;
} Float2;

/* Struct representing pointsi n 3-dimensional Euclidian space. */
typedef struct {
  float x;
  float y;
  float z;
} Float3;

Float2 add2f(Float2 a, Float2 b);

Float2 sub2f(Float2 a, Float2 b);
Float3 sub3f(Float3 a, Float3 b);

Float2 smul2f(float a, Float2 x);
Float3 smul3f(float a, Float3 x);

float dot2f(Float2 a, Float2 b);
float dot3f(Float3 a, Float3 b);

float norm2f(Float2 a);
float norm3f(Float3 a);

/* Integration rule for 1-dimensional integrals. 
 * 
 * nSamples - contains the number of sample points evaluated by this 
 *            integration rule.
 * pX - pointer/array with the abscissa values.
 * pW - pointer/array with the weights of the rule.
 */
typedef struct {
  int nSamples;     /* number of samples taken by integration rule */
  float *pX;  /* abscissa values of the integration rule     */
  float *pW;  /* weights of the integration rule             */
} IntRule1D;

/* Integration method for line segements in 2-dimensional space.
 * Parameters:
 *   integrand - a function pointer to the function being integrated along
 *               the line segment.
 *   state - an object containing user defined state data for the integrand function.
 *           Via this mechanism, the user of this method can pass additional state information
 *           to the integrand function.
 *   intRule - a 1-dimensional integration rule, which is being applied along the line segment.
 *   start - 2-dimensional start point of the line-segement being integrated along.
 *   end   - 2-dimensional end point of the line-segment being integrated along.
 */
complex float complexQuad2D(complex float(*integrand)(Float2, void*), void * state, IntRule1D intRule, Float2 start, Float2 end);

/* An array containing the abscissa values of the default 1-dimensional integration rule. */
extern float aX_1D[];
/* An array containing the weights of the default 1-dimensional integration rule. */
extern float aW_1D[];

/* Integration rule for 2-dimensional line integrals
 * 
 * nSamples - contains the number of sample points evaluated by this 
 *            integration rule.
 * pX - pointer/array with the x-values of the sample points.
 * pY - pointer/array with the y-values of the sample points.
 * pW - pointer/array with the weights of the rule.
 */
typedef struct {
  int nSamples; /* number of samples taken by integration rule */
  float *pX;    /* abscissa x-values of the integration rule     */
  float *pY;    /* abscissa y-values of the integration rule     */
  float *pW;    /* weights of the integration rule             */
  
} IntRule2D;

/* Integration method for triangles in 3-dimensional space.
 * Parameters:
 *   integrand - a function pointer to the function being integrated along
 *               the line segment.
 *   state - an object containing user defined state data for the integrand function.
 *           Via this mechanism, the user of this method can pass additional state information
 *           to the integrand function.
 *   intRule - a 1-dimensional integration rule, which is being applied along the line segment.
 *   a, b, c -   3-dimensional vertices/corners of the triangle.
 */
complex float complexQuad3D(complex float(*integrand)(Float3, void*), void * state, IntRule2D intRule,
			    Float3 a, Float3 b, Float3 c);

/* An array containing the x-abscissa values of the default 2-dimensional integration rule. */
extern float aX_2D[];
/* An array containing the y-abscissa values of the default 2-dimensional integration rule. */
extern float aY_2D[];
/* An array containing the weights of the default 1-dimensional integration rule. */
extern float aW_2D[];

/* Method for creating integration rules for the semi circle from [0, 2*PI].
 * 
 * Parameters:
 *   nSections - the method replicates the 1-dimensional integration rule provided
 *               by the next parameter (intRule) as many times as nSections along the
 *               semi-circle being integrated. This is essentially a way to control
 *               the total number of samples taken during integration, which is 
 *               nSections * intRule.nSamples.
 *   intRule - a 1-dimensional integration rule, which is used as the basis for the
 *             semi-circle integration rule being constructed.
 *   pSemiCircleRule - points to a rule struct that receives the newly generated 
 *                     semi-circle integration rule. NOTE: The pX, pY, and pW arrays 
 *                     must have sufficient length to receive the 
 *                             nSections * intRule.nSamples
 *                     number of samples generated. If the arrays are not of sufficient
 *                     size the method potentially overwrites other application data or
 *                     may cause a segmentation fault.
 */
void semiCircleIntegralRule(int nSections, IntRule1D intRule, IntRule2D * pSemiCircleRule);

/* Method for evaluating 2-dimensional line integrals.
 * 
 * Parameters:
 *   integrand - the function being integrated. 
 *   state - a user defined state struct that gets passed to the integrand function in addition to
 *           its 2-dimensional location parameter.
 *   intRule - the integration rule being evaluated. The integration rule essentially contains the 
 *             shape of the line being integrated along. 
 */
complex float complexLineIntegral(complex float(*integrand)(Float2, void*), void * state, IntRule2D intRule);
