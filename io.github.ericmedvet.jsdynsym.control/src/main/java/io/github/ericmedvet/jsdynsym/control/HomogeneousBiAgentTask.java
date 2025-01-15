package io.github.ericmedvet.jsdynsym.control;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Predicate;

public interface HomogeneousBiAgentTask<C extends DynamicalSystem<O, A, ?>, O, A, S>
    extends HomogeneousBiSimulation<C, HomogeneousBiAgentTask.Step<O, A, S>, Simulation.Outcome<HomogeneousBiAgentTask.Step<O, A, S>>> {

  record Step<O, A, S>(Pair<O, O> observations, Pair<A, A> actions, S state) {}

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> HomogeneousBiAgentTask<C, O, A, S> fromHomogenousBiEnvironment(
      HomogeneousBiEnvironment<O, A, S> biEnvironment,
      Pair<A, A> initialActions,
      Predicate<S> stopCondition,
      DoubleRange tRange,
      double dT
  ) {
    return (agent1, agent2) -> {
      biEnvironment.reset();
      agent1.reset();
      agent2.reset();
      double t = tRange.min();
      Map<Double, HomogeneousBiAgentTask.Step<O, A, S>> steps = new HashMap<>();
      Pair<O, O> observations = biEnvironment.step(t, initialActions);
      while (t <= tRange.max() && !stopCondition.test(biEnvironment.getState())) {
        Pair<A, A> actions = new Pair<>(agent1.step(t, observations.first()), agent2.step(t, observations.second()));
        observations = biEnvironment.step(t, actions);
        steps.put(t, new Step<>(observations, actions, biEnvironment.getState()));
      }
      return Outcome.of(new TreeMap<>(steps));
    };
  }
}
