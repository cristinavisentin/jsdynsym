/*-
 * ========================LICENSE_START=================================
 * jsdynsym-core
 * %%
 * Copyright (C) 2023 - 2025 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jsdynsym.core.numerical;

public class LinearAlgebraUtils {

  private LinearAlgebraUtils() {
  }

  public static double dotProduct(double[] v1, double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException(
          "Wrong (not equal) vector size: %d and %d".formatted(v1.length, v2.length)
      );
    }
    double sum = 0d;
    for (int i = 0; i < v1.length; i++) {
      sum += v1[i] * v2[i];
    }
    return sum;
  }

  public static double[] product(double[][] m, double[] v) {
    double[] o = new double[m.length];
    for (int j = 0; j < o.length; j++) {
      o[j] = dotProduct(m[j], v);
    }
    return o;
  }

  public static double[] sum(double[] v1, double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException(
          "Wrong (not equal) vector size: %d and %d".formatted(v1.length, v2.length)
      );
    }
    double[] sum = new double[v1.length];
    for (int i = 0; i < v1.length; i++) {
      sum[i] = v1[i] + v2[i];
    }
    return sum;
  }

}
