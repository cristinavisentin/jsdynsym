package io.github.ericmedvet.jsdynsym.control;

import io.github.ericmedvet.jnb.datastructure.Pair;

import java.util.SortedMap;

public interface BiSimulation<T1, T2, S, O extends Simulation.Outcome<S>> extends Simulation<Pair<T1, T2>, S, O>{

  O simulate(T1 t1, T2 t2);

  @Override
  default O simulate(Pair<T1, T2> tPair) {
    return simulate(tPair.first(), tPair.second());
  }
}
