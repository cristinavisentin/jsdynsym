/*-
 * ========================LICENSE_START=================================
 * jsdynsym-control
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
package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;

public class PongAgent implements MultivariateRealFunction {

  private final double maxNormalizedSpeed;

  public PongAgent() {
    maxNormalizedSpeed = 0.5;
  }

  public PongAgent(double normalizedSpeed) {
    this.maxNormalizedSpeed = normalizedSpeed;
  }

  // racketY, racketYV, ballX, ballY, ballXV, ballYV
  @Override
  public double[] compute(double... input) {
    double racketY = input[0];
    double racketYV = input[1];
    double ballY = input[3];
    if (Math.abs(racketYV) > maxNormalizedSpeed && racketYV * Math.signum(ballY - racketY) >= 0) {
      return new double[] {0};
    } else {
      return new double[] {Math.signum(ballY - racketY) * maxNormalizedSpeed};
    }
  }

  @Override
  public int nOfInputs() {
    return 6;
  }

  @Override
  public int nOfOutputs() {
    return 1;
  }

  @Override
  public String toString() {
    return "SimplePongAgent [maxNS=" + maxNormalizedSpeed + "]";
  }
}
