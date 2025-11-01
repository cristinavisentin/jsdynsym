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

import io.github.ericmedvet.jnb.datastructure.NumericalParametrized;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LinearCombination implements MultivariateRealFunction, NumericalParametrized<LinearCombination> {

  private final double[][] m;
  private final double[] q;
  private final boolean zeroQ;

  public LinearCombination(int nOfInputs, int nOfOutputs, boolean zeroQ) {
    this(
        IntStream.range(0, nOfOutputs)
            .mapToObj(i -> new double[nOfInputs])
            .toArray(
                double[][]::new
            ),
        new double[nOfOutputs],
        zeroQ
    );
  }

  public LinearCombination(double[][] m) {
    this(m, new double[m.length], true);
  }

  public LinearCombination(double[][] m, double[] q) {
    this(m, q, false);
  }

  private LinearCombination(double[][] m, double[] q, boolean zeroQ) {
    if (Arrays.stream(m).mapToInt(row -> row.length).distinct().count() != 1) {
      throw new IllegalArgumentException(
          "Wrong matrix size, %d columns with non-equal size: %s".formatted(
              m.length,
              Arrays.stream(m)
                  .map(row -> "%d".formatted(row.length))
                  .collect(Collectors.joining(", "))
          )
      );
    }
    if (m.length != q.length) {
      throw new IllegalArgumentException(
          "Wrong matrix/bias vector sizes: matrix is %dx%d, vector is %d and should be %d".formatted(
              m.length,
              m[0].length,
              q.length,
              m.length
          )
      );
    }
    this.m = m;
    this.zeroQ = zeroQ;
    if (!zeroQ) {
      this.q = q;
    } else {
      this.q = new double[m.length];
    }
  }

  @Override
  public double[] compute(double... input) {
    if (input.length != nOfInputs()) {
      throw new IllegalArgumentException(
          "Wrong input size: %d found, %d expected".formatted(
              input.length,
              nOfInputs()
          )
      );
    }
    return LinearAlgebraUtils.sum(LinearAlgebraUtils.product(m, input), q);
  }

  @Override
  public double[] getParams() {
    int nOfInputs = nOfInputs();
    int nOfOutputs = nOfOutputs();
    double[] params = new double[nOfInputs * nOfOutputs + (zeroQ ? 0 : nOfOutputs)];
    int c = 0;
    for (int i = 0; i < nOfOutputs; i = i + 1) {
      for (int j = 0; j < nOfInputs; j = j + 1) {
        params[c++] = m[i][j];
      }
    }
    if (!zeroQ) {
      for (int j = 0; j < nOfOutputs; j = j + 1) {
        params[c++] = q[j];
      }
    }
    return params;
  }

  @Override
  public void setParams(double[] params) {
    int nOfInputs = nOfInputs();
    int nOfOutputs = nOfOutputs();
    if (params.length != nOfInputs * nOfOutputs + (zeroQ ? 0 : nOfOutputs)) {
      throw new IllegalArgumentException(
          "Wrong flat params size: %d found, %dx%d+%d=%d expected".formatted(
              params.length,
              nOfInputs,
              nOfOutputs,
              nOfOutputs,
              nOfInputs * nOfOutputs + nOfOutputs
          )
      );
    }
    int c = 0;
    for (int i = 0; i < nOfOutputs; i = i + 1) {
      for (int j = 0; j < nOfInputs; j = j + 1) {
        m[i][j] = params[c++];
      }
    }
    if (zeroQ) {
      for (int j = 0; j < nOfOutputs; j = j + 1) {
        q[j] = params[c++];
      }
    }
  }

  @Override
  public int nOfInputs() {
    return m[0].length;
  }

  @Override
  public int nOfOutputs() {
    return m.length;
  }
}
