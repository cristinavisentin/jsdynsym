package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;

public class SimplePongAgent implements MultivariateRealFunction {

  // racketY, racketYV, ballX, ballY, ballXV, ballYV
  @Override
  public double[] compute(double... input) {
    double racketY = input[0];
    double ballY = input[3];
    return new double[] {ballY - racketY};
  }

  @Override
  public int nOfInputs() {
    return 6;
  }

  @Override
  public int nOfOutputs() {
    return 1;
  }
}