package io.github.ericmedvet.jsdynsym.control;

public interface HomogeneousBiSimulationWithExample<T, S, O extends Simulation.Outcome<S>> extends HomogeneousBiSimulation<T, S, O> {
  T example();
}
