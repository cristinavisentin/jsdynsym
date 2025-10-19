/*-
 * ========================LICENSE_START=================================
 * jsdynsym-experimenter
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
package io.github.ericmedvet.jsdynsym;

import io.github.ericmedvet.jnb.core.NamedBuilder;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Grid;
import io.github.ericmedvet.jnb.datastructure.NamedFunction;
import io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems;
import io.github.ericmedvet.jsdynsym.control.Environment;
import io.github.ericmedvet.jsdynsym.control.Simulation.Outcome;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask.Step;
import io.github.ericmedvet.jsdynsym.control.SingleRLAgentTask;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationDrawer;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationDrawer.Configuration;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment.State;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.rl.LinearActorCritic;
import io.github.ericmedvet.jsdynsym.core.rl.NumericalReinforcementLearningAgent;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent.RewardedInput;
import io.github.ericmedvet.jviz.core.plot.Value;
import io.github.ericmedvet.jviz.core.plot.XYDataSeries;
import io.github.ericmedvet.jviz.core.plot.XYDataSeries.Point;
import io.github.ericmedvet.jviz.core.plot.XYDataSeriesPlot;
import io.github.ericmedvet.jviz.core.plot.XYPlot.TitledData;
import io.github.ericmedvet.jviz.core.plot.image.LinesPlotDrawer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.SequencedMap;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RLExperiment {


  public static void main(String[] args) {
    NamedBuilder<?> nb = NamedBuilder.fromDiscovery();
    // task and reward
    @SuppressWarnings("unchecked") Environment<double[], double[], State, NumericalDynamicalSystem<?>> environment = (Environment<double[], double[], NavigationEnvironment.State, NumericalDynamicalSystem<?>>) nb
        .build(
            """
                ds.e.navigation(
                  arena = ds.arena.prepared(
                    name = u_barrier;
                    initialRobotXRange = m.range(min=0.45;max=0.55);
                    initialRobotYRange = m.range(min=0.7;max=0.8)
                  );
                  relativeV = true;
                  robotMaxV = 0.1;
                  robotRadius = 0.01
                )
                """
        );
    double targetProximityRadius = 0.1;
    double targetProximityReward = 1;
    double collisionPenalty = 0.01;
    SingleRLAgentTask<ReinforcementLearningAgent<double[], double[], ?>, double[], double[], State> rlTask = SingleRLAgentTask
        .fromEnvironment(
            () -> environment,
            environment.defaultObservation(),
            null,
            s -> false,
            false,
            (s, a) -> {
              double currentDistance = s.robotPosition().distance(s.targetPosition());
              double previousDistance = s.robotPreviousPosition().distance(s.targetPosition());
              double reward = previousDistance - currentDistance;
              reward = reward + (currentDistance < targetProximityRadius ? targetProximityReward : 0d);
              reward = reward + (s.hasCollided() ? collisionPenalty : 0d);
              return reward;
            }
        );
    // drawer
    NavigationDrawer drawer = new NavigationDrawer(Configuration.DEFAULT);
    // non learning agent
    @SuppressWarnings("unchecked") MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>) nb
        .build("ds.num.mlp(innerLayers = [16; 16])"))
        .apply(environment.exampleAgent().nOfInputs(), environment.exampleAgent().nOfOutputs());
    mlp.randomize(new Random(2), DoubleRange.SYMMETRIC_UNIT);
    NumericalReinforcementLearningAgent<?> mlpAgent = NumericalReinforcementLearningAgent.from(
        mlp
    );
    NumericalReinforcementLearningAgent<?> actorCritic = new LinearActorCritic(
        mlpAgent.nOfInputs(),
        mlpAgent.nOfOutputs(),
        0.0005,
        0.005,
        0.975,
        0.1,
        new DoubleRange(-0.1, 0.1),
        RandomGenerator.getDefault()
    );
    SequencedMap<String, Supplier<NumericalReinforcementLearningAgent<?>>> namedAgents = new LinkedHashMap<>(
        Map.ofEntries(
            Map.entry(
                "ac-0005",
                () -> new LinearActorCritic(
                    mlpAgent.nOfInputs(),
                    mlpAgent.nOfOutputs(),
                    0.0005,
                    0.005,
                    0.975,
                    0.1,
                    new DoubleRange(-0.1, 0.1),
                    RandomGenerator.getDefault()
                )
            ),
            Map.entry(
                "ac-001",
                () -> new LinearActorCritic(
                    mlpAgent.nOfInputs(),
                    mlpAgent.nOfOutputs(),
                    0.001,
                    0.005,
                    0.975,
                    0.1,
                    new DoubleRange(-0.1, 0.1),
                    RandomGenerator.getDefault()
                )
            ),
            Map.entry(
                "ac-005",
                () -> new LinearActorCritic(
                    mlpAgent.nOfInputs(),
                    mlpAgent.nOfOutputs(),
                    0.005,
                    0.005,
                    0.975,
                    0.1,
                    new DoubleRange(-0.1, 0.1),
                    RandomGenerator.getDefault()
                )
            )
        )
    );
    List<NamedFunction<Outcome<Step<RewardedInput<double[]>, double[], NavigationEnvironment.State>>, Double>> functions = List
        .of(
            NamedFunction.from(
                o -> o.snapshots().values().stream().mapToDouble(s -> s.observation().reward()).sum(),
                "cumulated.reward"
            ),
            NamedFunction.from(
                o -> o.snapshots()
                    .lastEntry()
                    .getValue()
                    .state()
                    .robotPosition()
                    .distance(o.snapshots().lastEntry().getValue().state().targetPosition()),
                "final.dist"
            )
        );
    int nOfEpisodes = 1000;
    double dT = 0.1;
    double finalT = 60;
    Map<String, List<Outcome<Step<RewardedInput<double[]>, double[], NavigationEnvironment.State>>>> outcomes = namedAgents
        .keySet()
        .stream()
        .collect(
            Collectors.toMap(
                Function.identity(),
                name -> {
                  NumericalReinforcementLearningAgent<?> agent = namedAgents.get(name).get();
                  agent.reset();
                  return runEpisodes(rlTask, agent, dT, finalT, nOfEpisodes);
                }
            )
        );
    Grid<TitledData<List<XYDataSeries>>> dataGrid = Grid.create(
        functions.size(),
        1,
        (fI, pI) -> new TitledData<>(
            functions.get(fI).name(),
            "problem",
            namedAgents.keySet()
                .stream()
                .map(name -> {
                  double sumOfSteps = 0;
                  List<Point> points = new ArrayList<>(outcomes.get(name).size());
                  for (Outcome<Step<RewardedInput<double[]>, double[], NavigationEnvironment.State>> outcome : outcomes
                      .get(name)) {
                    sumOfSteps = sumOfSteps + outcome.snapshots().size();
                    points.add(
                        new Point(
                            Value.of(sumOfSteps),
                            Value.of(functions.get(fI).apply(outcome))
                        )
                    );
                  }
                  return XYDataSeries.of(
                      name,
                      points
                  );
                }
                )
                .toList()
        )
    );
    new LinesPlotDrawer(io.github.ericmedvet.jviz.core.plot.image.Configuration.FREE_SCALES).show(
        new XYDataSeriesPlot(
            "RL experiment",
            "Metric",
            "Problem",
            "N. of steps",
            "Value",
            DoubleRange.UNBOUNDED,
            DoubleRange.UNBOUNDED,
            dataGrid
        )
    );
  }

  static <O, A, S> Outcome<Step<O, A, S>> translate(Outcome<Step<RewardedInput<O>, A, S>> outcome) {
    return Outcome.of(
        new TreeMap<>(
            outcome.snapshots()
                .entrySet()
                .stream()
                .collect(
                    Collectors.toMap(
                        Entry::getKey,
                        e -> new Step<>(
                            e.getValue().observation().input(),
                            e.getValue().action(),
                            e.getValue().state()
                        )
                    )
                )
        )
    );
  }

  private static <O, A, S> List<Outcome<Step<RewardedInput<O>, A, S>>> runEpisodes(
      SingleRLAgentTask<ReinforcementLearningAgent<O, A, ?>, O, A, S> rlTask,
      ReinforcementLearningAgent<O, A, ?> rlAgent,
      double dT,
      double finalT,
      int nOfEpisodes
  ) {
    return IntStream.range(0, nOfEpisodes)
        .mapToObj(i -> rlTask.simulate(rlAgent, dT, new DoubleRange(0, finalT)))
        .toList();
  }

}
