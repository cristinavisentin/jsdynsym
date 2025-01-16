package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;

public class SimplePongAgent implements MultivariateRealFunction {

  private final double dumpingEffect;

  public SimplePongAgent() {
    dumpingEffect = 0.5;
  }

  public SimplePongAgent(double dumpingEffect) {
    this.dumpingEffect = dumpingEffect;
  }

  // racketY, racketYV, ballX, ballY, ballXV, ballYV
  @Override
  public double[] compute(double... input) {
    double racketY = input[0];
    double ballY = input[3];
    return new double[] {(ballY - racketY) * dumpingEffect};
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