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

  private final double normalizedDeltaY;

  public PongAgent() {
    normalizedDeltaY = 0.5;
  }

  public PongAgent(double normalizedDeltaY) {
    this.normalizedDeltaY = normalizedDeltaY;
  }

  @Override
  public double[] compute(double... input) { // racketY, ballX, ballY, ballVX, ballVY
    double racketY = input[0];
    double ballY = input[2];
    double diff = ballY - racketY;
    return new double[]{Math.signum(diff) * normalizedDeltaY};
  }

  @Override
  public int nOfInputs() {
    return 5;
  }

  @Override
  public int nOfOutputs() {
    return 1;
  }
}
