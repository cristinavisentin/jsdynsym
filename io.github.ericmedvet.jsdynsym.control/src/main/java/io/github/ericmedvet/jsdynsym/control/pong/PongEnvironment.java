package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.control.SymmetricBiEnvironment;
import io.github.ericmedvet.jsdynsym.control.geometry.Point;

import java.util.random.RandomGenerator;

public class PongEnvironment implements SymmetricBiEnvironment<double[], double[], PongEnvironment.State> {

  public record Configuration(
      DoubleRange racketsInitialYRange,
      double racketsLength,
      double racketsMaxYVelocity,
      double ballRadius,
      DoubleRange ballInitialXVelocityRange,
      DoubleRange ballInitialYVelocityRange,
      double arenaWidth,
      double arenaHeight,
      RandomGenerator randomGenerator
  ) {}

  public record State(
      Configuration configuration,
      double firstRacketY,
      double secondRacketY,
      Point ballPosition,
      Point ballVelocity,
      double firstRacketYVelocity,
      double secondRacketYVelocity,
      int firstRacketScore,
      int secondRacketScore
  ) {}

  @Override
  public double[] defaultAgent1Action() {
    return new double[0];
  }

  @Override
  public double[] defaultAgent2Action() {
    return new double[0];
  }

  @Override
  public State getState() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public Pair<double[], double[]> step(double t, Pair<double[], double[]> input) {
    return null;
  }
}
