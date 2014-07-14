/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.util

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite
import org.apache.spark.mllib.util.TestingUtils._

class TestingUtilsSuite extends FunSuite {

  test("implicit class DoubleWithAlmostEquals") {
    // Regular large numbers - generally not problematic
    assert(1000000.0.almostEquals(1000001.0, 1E-5))
    assert(1000001.0.almostEquals(1000000.0, 1E-5))
    assert(!10000.0.almostEquals(10001.0, 1E-5))
    assert(!10001.0.almostEquals(10000.0, 1E-5))
    assert((-1000000.0).almostEquals(-1000001.0, 1E-5))
    assert((-1000001.0).almostEquals(-1000000.0, 1E-5))
    assert(!(-10000.0).almostEquals(-10001.0, 1E-5))
    assert(!(-10001.0).almostEquals(-10000.0, 1E-5))

    // Numbers around 1, and -1
    assert(1.0000001.almostEquals(1.0000002, 1E-5))
    assert(1.0000002.almostEquals(1.0000001, 1E-5))
    assert(!1.0002.almostEquals(1.0001, 1E-5))
    assert(!1.0001.almostEquals(1.0002, 1E-5))
    assert((-1.000001).almostEquals(-1.000002, 1E-5))
    assert((-1.000002).almostEquals(-1.000001, 1E-5))
    assert(!(-1.0001).almostEquals(-1.0002, 1E-5))
    assert(!(-1.0002).almostEquals(-1.0001, 1E-5))

    // Numbers between 1 and 0, and 0 and -1
    assert(0.000000001000001.almostEquals(0.000000001000002, 1E-5))
    assert(0.000000001000002.almostEquals(0.000000001000001, 1E-5))
    assert(!0.000000000001002.almostEquals(0.000000000001001, 1E-5))
    assert(!0.000000000001001.almostEquals(0.000000000001002, 1E-5))
    assert((-0.000000001000001).almostEquals(-0.000000001000002, 1E-5))
    assert((-0.000000001000002).almostEquals(-0.000000001000001, 1E-5))
    assert(!(-0.000000000001002).almostEquals(-0.000000000001001, 1E-5))
    assert(!(-0.000000000001001).almostEquals(-0.000000000001002, 1E-5))

    // Comparisons involving zero The absolute error will
    // be used instead since the relative error is meaningless.
    assert(0.0.almostEquals(0.0, 1E-5))
    assert(0.0.almostEquals(-0.0f, 1E-5))
    assert((-0.0).almostEquals(-0.0, 1E-5))
    assert(0.00000001.almostEquals(0.0, 1E-5))
    assert(0.0.almostEquals(0.00000001, 1E-5))
    assert(!0.0.almostEquals(0.0001, 1E-5))
    assert(!0.0001.almostEquals(0.0, 1E-5))
    assert((-0.00000001).almostEquals(0.0, 1E-5))
    assert(0.0.almostEquals(-0.00000001, 1E-5))
    assert(0.0.almostEquals(1E-40))
    assert(1E-40.almostEquals(0.0))
    assert(0.0.almostEquals(-1e-40))
    assert((-1E-40).almostEquals(0.0))

    // Comparisons involving infinities, and NaN
    assert(Double.PositiveInfinity.almostEquals(Double.PositiveInfinity))
    assert(Double.NegativeInfinity.almostEquals(Double.NegativeInfinity))
    assert(!Double.PositiveInfinity.almostEquals(Double.NegativeInfinity))
    assert(!Double.PositiveInfinity.almostEquals(Double.MaxValue))
    assert(!Double.NegativeInfinity.almostEquals(-Double.MaxValue))
    assert(!Double.NaN.almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(0.0))
    assert(!(-0.0).almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(-0.0))
    assert(!0.0.almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(Double.PositiveInfinity))
    assert(!Double.PositiveInfinity.almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(Double.NegativeInfinity))
    assert(!Double.NegativeInfinity.almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(Double.MaxValue))
    assert(!Double.MaxValue.almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(-Double.MaxValue))
    assert(!(-Double.MaxValue).almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(Double.MinValue))
    assert(!Double.MinValue.almostEquals(Double.NaN))
    assert(!Double.NaN.almostEquals(-Double.MinValue))
    assert(!(-Double.MinValue).almostEquals(Double.NaN))

    // Comparisons of numbers very close to zero.
    assert((10 * Double.MinPositiveValue).almostEquals(10 * -Double.MinPositiveValue))
    assert((10000 * Double.MinPositiveValue).almostEquals(10000 * -Double.MinPositiveValue))
    assert(Double.MinPositiveValue.almostEquals(-Double.MinPositiveValue))
    assert((-Double.MinPositiveValue).almostEquals(Double.MinPositiveValue))
    assert(Double.MinPositiveValue.almostEquals(0))
    assert(0.0.almostEquals(Double.MinPositiveValue))
    assert((-Double.MinPositiveValue).almostEquals(0.0))
    assert(0.0.almostEquals(-Double.MinPositiveValue))
    assert(1E-12.almostEquals(-Double.MinPositiveValue))
    assert(1E-12.almostEquals(Double.MinPositiveValue))
    assert(Double.MinPositiveValue.almostEquals(1E-12))
    assert((-Double.MinPositiveValue).almostEquals(1E-12))
  }

  test("implicit class VectorWithAlmostEquals") {
    // Comparisons of two dense vectors
    assert(Vectors.dense(Array(3.1, 3.5, 0.0)).almostEquals(
      Vectors.dense(Array(3.1 + 1E-8, 3.5 + 2E-7, 1E-8)), 1E-6))
    assert(Vectors.dense(Array(3.1 + 1E-8, 3.5 + 2E-7, 1E-8)).almostEquals(
      Vectors.dense(Array(3.1, 3.5, 0.0)), 1E-6))

    assert(!Vectors.dense(Array(3.1, 3.5, 0.0)).almostEquals(
      Vectors.dense(Array(3.1 + 1E-5, 3.5 + 2E-7, 1E-3)), 1E-6))
    assert(!Vectors.dense(Array(3.1 + 1E-5, 3.5 + 2E-7, 1E-3)).almostEquals(
      Vectors.dense(Array(3.1, 3.5, 0.0)), 1E-6))

    // Comparisons of two sparse vectors
    assert(Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)).almostEquals(
      Vectors.sparse(3, Array(0, 2), Array(3.1 + 1E-8, 2.4 + 1E-7)), 1E-6))
    assert(Vectors.sparse(3, Array(0, 2), Array(3.1 + 1E-8, 2.4 + 1E-7)).almostEquals(
      Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)), 1E-6))

    assert(!Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)).almostEquals(
      Vectors.sparse(3, Array(0, 2), Array(3.1 + 1E-3, 2.4)), 1E-6))
    assert(!Vectors.sparse(3, Array(0, 2), Array(3.1 + 1E-3, 2.4)).almostEquals(
      Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)), 1E-6))

    // Comparisons of a dense vector and a sparse vector
    assert(Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)).almostEquals(
      Vectors.dense(Array(3.1 + 1E-8, 0, 2.4 + 1E-7)), 1E-6))
    assert(Vectors.dense(Array(3.1 + 1E-8, 0, 2.4 + 1E-7)).almostEquals(
      Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)), 1E-6))

    assert(!Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)).almostEquals(
      Vectors.dense(Array(3.1, 1E-3, 2.4)), 1E-6))
    assert(!Vectors.dense(Array(3.1 + 1E-3, 0, 2.4)).almostEquals(
      Vectors.sparse(3, Array(0, 2), Array(3.1, 2.4)), 1E-6))
  }
}
