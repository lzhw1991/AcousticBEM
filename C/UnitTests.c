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
 * along with Foobar.  If not, see <http://www.gnu.org/licenses/>.             *
 * --------------------------------------------------------------------------- */
#include <stdio.h>
#include <math.h>

#include "cutest/CuTest.h"
#include "Operators.h"


complex float constantOne2D(Float2 p, void * pState) {
  return 1.0f + I * 0.0f;
}

complex float constantOne3D(Float3 p, void * pState) {
  return 1.0f + I * 0.0f;
}

complex float constantTwo2D(Float2 p, void * pState) {
  return 2.0f + I * 0.0f;
}

void TestComplexQuad2D(CuTest *tc) {
  IntRule1D rule = {8, aX_1D, aW_1D};
  
  Float2 start = {0.0f, 0.0f};
  Float2 end   = {1.0f, 1.0f};
  
  complex float result = complexQuad2D(constantOne2D, (void*)0, rule, start, end);
  CuAssertComplexFloatEquals(tc, sqrt(2.0f), result, 1e-7f);
}

void TestComplexQuad3D(CuTest *tc) {
  IntRule2D rule = {7, aX_2D, aY_2D, aW_2D};
  
  Float3 a = {0.0f, 0.0f, 0.0f};
  Float3 b = {1.0f, 1.0f, 1.0f};
  Float3 c = {0.0f, 1.0f, 1.0f};
  
  complex float result = complexQuad3D(constantOne3D, (void*)0, rule, a, b, c);
  CuAssertComplexFloatEquals(tc, sqrt(0.5f), result, 1e-7);
}

void TestSemiCircleIntegralRule01(CuTest *tc) {
  IntRule1D rule = {8, aX_1D, aW_1D};
  float circleX[8];
  float circleY[8];
  float circleW[8];
  IntRule2D circleRule = {0, circleX, circleY, circleW};

  semiCircleIntegralRule(1, rule, &circleRule);
  CuAssertIntEquals(tc, 8, circleRule.nSamples);
  for (int i = 0; i < circleRule.nSamples; ++i) {
    Float2 p = {circleRule.pX[i], circleRule.pY[i]};
    CuAssertFloatEquals(tc, 1.0f, norm2f(p), 1e-7);
  }
}

void TestSemiCircleIntegralRule02(CuTest *tc) {
  IntRule1D rule = {8, aX_1D, aW_1D};
  float circleX[16];
  float circleY[16];
  float circleW[16];
  IntRule2D circleRule = {0, circleX, circleY, circleW};

  semiCircleIntegralRule(2, rule, &circleRule);
  CuAssertIntEquals(tc, 16, circleRule.nSamples);
  for (int i = 0; i < circleRule.nSamples; ++i) {
    Float2 p = {circleRule.pX[i], circleRule.pY[i]};
    CuAssertFloatEquals(tc, 1.0f, norm2f(p), 1e-7);
  }
}

void TestSemiCircleLineIntegral01(CuTest *tc) {
  IntRule1D rule = {8, aX_1D, aW_1D};
  float circleX[16];
  float circleY[16];
  float circleW[16];
  IntRule2D circleRule = {0, circleX, circleY, circleW};

  semiCircleIntegralRule(1, rule, &circleRule);

  complex float result = complexLineIntegral(constantOne2D, (void *)0, circleRule);
  CuAssertFloatEquals(tc, M_PI, result, 1e-7);
}

void TestSemiCircleLineIntegral02(CuTest *tc) {
  IntRule1D rule = {8, aX_1D, aW_1D};
  float circleX[16];
  float circleY[16];
  float circleW[16];
  IntRule2D circleRule = {0, circleX, circleY, circleW};

  semiCircleIntegralRule(2, rule, &circleRule);

  complex float result = complexLineIntegral(constantOne2D, (void *)0, circleRule);
  CuAssertFloatEquals(tc, M_PI, result, 1e-6);
}

void TestSemiCircleLineIntegral03(CuTest *tc) {
  IntRule1D rule = {8, aX_1D, aW_1D};
  float circleX[16];
  float circleY[16];
  float circleW[16];
  IntRule2D circleRule = {0, circleX, circleY, circleW};

  semiCircleIntegralRule(2, rule, &circleRule);

  complex float result = complexLineIntegral(constantTwo2D, (void *)0, circleRule);
  CuAssertFloatEquals(tc, 2.0f * M_PI, result, 1e-6);
}

CuSuite* IntegralTestsGetSuite() {
  CuSuite* suite = CuSuiteNew();
  SUITE_ADD_TEST(suite, TestComplexQuad2D);
  SUITE_ADD_TEST(suite, TestComplexQuad3D);
  SUITE_ADD_TEST(suite, TestSemiCircleIntegralRule01); 
  SUITE_ADD_TEST(suite, TestSemiCircleIntegralRule02);
  SUITE_ADD_TEST(suite, TestSemiCircleLineIntegral01);
  SUITE_ADD_TEST(suite, TestSemiCircleLineIntegral02);
  SUITE_ADD_TEST(suite, TestSemiCircleLineIntegral03);
 return suite;
}

void Test_smul3f(CuTest *tc) {
  Float3 x = {1.0f, 2.0f, 3.0f};
  x = smul3f(0.5f, x);
  CuAssertFloatEquals(tc, 0.5f, x.x, 1e-7);
  CuAssertFloatEquals(tc, 1.0f, x.y, 1e-7);
  CuAssertFloatEquals(tc, 1.5f, x.z, 1e-7);
}

CuSuite* GeometryTestsGetSuite() {
  CuSuite* suite = CuSuiteNew();
  SUITE_ADD_TEST(suite, Test_smul3f);
 return suite;
}

void RunAllTests(void) {
  CuString *output = CuStringNew();
  CuSuite* suite = CuSuiteNew();
  
  CuSuiteAddSuite(suite, IntegralTestsGetSuite());
  CuSuiteAddSuite(suite, GeometryTestsGetSuite());
  
  CuSuiteRun(suite);
  CuSuiteSummary(suite, output);
  CuSuiteDetails(suite, output);
  printf("%s\n", output->buffer);
}

int main(void) {
  RunAllTests();
}
