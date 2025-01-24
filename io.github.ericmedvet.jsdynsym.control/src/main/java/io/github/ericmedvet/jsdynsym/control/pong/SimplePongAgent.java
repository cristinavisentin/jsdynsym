package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;

public class SimplePongAgent implements MultivariateRealFunction {

  private final double normalizedDeltaY;

  public SimplePongAgent() {
    normalizedDeltaY = 0.5;
  }

  public SimplePongAgent(double normalizedDeltaY) {
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
