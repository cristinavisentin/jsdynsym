package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;

public class DummyPongAgent implements MultivariateRealFunction {

  @Override
  public double[] compute(double... input) {
    return new double[] {0};
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
