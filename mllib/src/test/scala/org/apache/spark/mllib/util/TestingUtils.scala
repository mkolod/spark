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

import org.apache.spark.mllib.linalg.Vector

object TestingUtils {

  implicit class DoubleWithAlmostEquals(val x: Double) {
    def almostEquals(y: Double, eps: Double = 1E-6): Boolean = {
      val absX = math.abs(x)
      val absY = math.abs(y)
      val diff = math.abs(x - y)
      if (x == y) {
        true
      } else if (absX < 1E-15 || absY < 1E-15) {
        // x or y is zero or extremely close to it; the relative error is meaningless here.
        // As a result, we do absolute error comparison instead.
        diff < eps
      } else {
        diff < eps * math.min(absX, absY)
      }
    }
  }

  implicit class VectorWithAlmostEquals(val x: Vector) {
    def almostEquals(y: Vector, eps: Double = 1E-6): Boolean =
      x.toArray.zip(y.toArray).forall(x => x._1.almostEquals(x._2, eps))
  }

}
