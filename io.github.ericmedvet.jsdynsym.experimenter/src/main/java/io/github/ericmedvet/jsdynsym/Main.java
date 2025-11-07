/*-
 * ========================LICENSE_START=================================
 * jsdynsym-buildable
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
import io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.Builder;
import io.github.ericmedvet.jsdynsym.control.Environment;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.control.navigation.*;
import io.github.ericmedvet.jsdynsym.core.numerical.LinearCombination;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.HebbianMultilayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import io.github.ericmedvet.jviz.core.drawer.Drawer;
import io.github.ericmedvet.jviz.core.drawer.Drawer.Arrangement;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.DoubleStream;

public class Main {

  public static void main(String[] args) throws IOException {
    // navigation();
    // pointNavigation();
    hebbianNavigation();
  }

  @SuppressWarnings("unchecked")
  public static void pointNavVisual() {
    NamedBuilder<?> nb = NamedBuilder.fromDiscovery();
    String genotype = "rO0ABXNyABNqYXZhLnV0aWwuQXJyYXlMaXN0eIHSHZnHYZ0DAAFJAARzaXpleHAAAAAWdwQAAAAWc3IAEGphdmEubGFuZy5Eb3VibGWAs8JKKWv7BAIAAUQABXZhbHVleHIAEGphdmEubGFuZy5OdW1iZXKGrJUdC5TgiwIAAHhwP+HfAxn4soBzcQB+AAI/5/cWGhuXynNxAH4AAr/VPpwJHdyAc3EAfgACv8jBAcrQY6BzcQB+AAK/5cwX9vm3MnNxAH4AAj/desmMrqzYc3EAfgACv8/9cWhZJnhzcQB+AAK/z1669BZtQHNxAH4AAj+/BbKVlmMwc3EAfgACv+biExNRExBzcQB+AAI/2NINXFEfoHNxAH4AAr/n8MgDN+0Mc3EAfgACP8gJ2fyu00hzcQB+AAK/7FiZq3ZpsHNxAH4AAr/sodjxwbdqc3EAfgACP8oN6i+X/JhzcQB+AAK/zzJxehFIqHNxAH4AAr/mXy4s9PQic3EAfgACP4xC3kyxYYBzcQB+AAI/4yais6r7EHNxAH4AAj/YYcjqIOGgc3EAfgACP+HNfeh3wOZ4";
    Function<String, Object> decoder = (Function<String, Object>) nb.build("f.fromBase64()");
    List<Double> actualGenotype = (List<Double>) decoder.apply(genotype);
    PointNavigationEnvironment environment = (PointNavigationEnvironment) nb.build(
        "ds.e.pointNavigation(arena = E_MAZE;initialRobotXRange = m.range(min = 0.5; max = 0.55);" + "initialRobotYRange = m.range(min = 0.75; max = 0.75);robotMaxV = 0.05)"
    );
    MultiLayerPerceptron mlp = ((Builder<MultiLayerPerceptron, ?>) nb.build(
        "ds.num.mlp(innerLayerRatio = 2.0)"
    ))
        .apply(environment.nOfOutputs(), environment.nOfInputs());
    mlp.setParams(actualGenotype.stream().mapToDouble(d -> d).toArray());
    SingleAgentTask<NumericalDynamicalSystem<?>, double[], double[], PointNavigationEnvironment.State> task = SingleAgentTask
        .fromEnvironment(
            () -> environment,
            s -> s.robotPosition().distance(s.targetPosition()) < .01,
            true
        );
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>> outcome = task
        .simulate(mlp, 0.1, new DoubleRange(0, 100));
    PointNavigationDrawer d = new PointNavigationDrawer(
        PointNavigationDrawer.Configuration.DEFAULT
    );
    d.show(new Drawer.ImageInfo(500, 500), outcome);
    Function<Simulation.Outcome<SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>>, Double> fitness = (Function<Simulation.Outcome<SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>>, Double>) nb
        .build("ds.e.n.finalTimePlusD()");
    System.out.println(fitness.apply(outcome));
    /*VectorFieldDrawer vfd =
    new VectorFieldDrawer(Arena.Prepared.E_MAZE.arena(), VectorFieldDrawer.Configuration.DEFAULT);
    vfd.show(new ImageBuilder.ImageInfo(500, 500), mlp);*/
  }

  public static void pointNavigation() {
    NamedBuilder<?> nb = NamedBuilder.fromDiscovery();
    PointNavigationEnvironment environment = (PointNavigationEnvironment) nb.build(
        "ds.e.pointNavigation(arena = E_MAZE)"
    );
    @SuppressWarnings("unchecked") MultiLayerPerceptron mlp = ((Builder<MultiLayerPerceptron, ?>) nb
        .build("ds.num.mlp()"))
        .apply(environment.nOfOutputs(), environment.nOfInputs());
    mlp.randomize(new Random(), DoubleRange.SYMMETRIC_UNIT);
    VectorFieldDrawer vfd = new VectorFieldDrawer(
        Arena.Prepared.E_MAZE.arena(),
        VectorFieldDrawer.Configuration.DEFAULT
    );
    vfd.show(new Drawer.ImageInfo(500, 500), mlp);
    SingleAgentTask<NumericalDynamicalSystem<?>, double[], double[], PointNavigationEnvironment.State> task = SingleAgentTask
        .fromEnvironment(() -> environment, s -> false, true);
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>> outcome = task
        .simulate(mlp, 0.1, new DoubleRange(0, 10));
    new PointNavigationDrawer(PointNavigationDrawer.Configuration.DEFAULT)
        .videoBuilder()
        .save(new File("../point-navigation.mp4"), outcome);
  }

  public static void navigation() {
    NamedBuilder<?> nb = NamedBuilder.fromDiscovery();
    @SuppressWarnings("unchecked") Environment<double[], double[], NavigationEnvironment.State, NumericalDynamicalSystem<?>> environment = (Environment<double[], double[], NavigationEnvironment.State, NumericalDynamicalSystem<?>>) nb
        .build(
            """
                ds.e.navigation(
                  arena = ds.arena.prepared(
                    name = u_barrier;
                    initialRobotXRange = m.range(min=0.5;max=0.5);
                    initialRobotYRange = m.range(min=0.8;max=0.8)
                  );
                  relativeV = true;
                  robotMaxV = 0.1;
                  robotRadius = 0.01
                )
                """
        );
    @SuppressWarnings("unchecked") MultiLayerPerceptron mlp = ((Builder<MultiLayerPerceptron, ?>) nb
        .build("ds.num.mlp(innerLayers = [16; 16])"))
        .apply(environment.exampleAgent().nOfInputs(), environment.exampleAgent().nOfOutputs());
    mlp.randomize(new Random(2), DoubleRange.SYMMETRIC_UNIT);
    LinearCombination linear = new LinearCombination(mlp.nOfInputs(), mlp.nOfOutputs(), false);
    linear.randomize(new Random(2), DoubleRange.SYMMETRIC_UNIT);
    NumericalDynamicalSystem<?> agent = linear;
    SingleAgentTask<NumericalDynamicalSystem<?>, double[], double[], NavigationEnvironment.State> task = SingleAgentTask
        .fromEnvironment(() -> environment, s -> false, true);
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>> outcome = task.simulate(
        agent,
        1,
        new DoubleRange(0, 30)
    );
    NavigationDrawer d = new NavigationDrawer(NavigationDrawer.Configuration.DEFAULT);
    @SuppressWarnings("unchecked") Function<Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>, Double> fitness = (Function<Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>, Double>) nb
        .build("ds.e.n.arenaCoverage()");
    System.out.println(fitness.apply(outcome));
    Function<Double, Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>> tResF = dT -> SingleAgentTask
        .fromEnvironment(
            () -> environment,
            s -> false,
            true
        )
        .simulate(agent, dT, new DoubleRange(0, 30));
    d.multi(Arrangement.HORIZONTAL)
        .show(
            DoubleStream.iterate(0.05, v -> v <= 0.25, v -> v + 0.10).boxed().map(tResF).toList()
        );
  }

  public static void hebbianNavigation() {
    NamedBuilder<?> nb = NamedBuilder.fromDiscovery();
    @SuppressWarnings("unchecked") Environment<double[], double[], NavigationEnvironment.State, NumericalDynamicalSystem<?>> environment = (Environment<double[], double[], NavigationEnvironment.State, NumericalDynamicalSystem<?>>) nb
        .build(
            """
                ds.e.navigation(
                  arena = ds.arena.prepared(
                    name = u_barrier;
                    initialRobotXRange = m.range(min=0.5;max=0.5);
                    initialRobotYRange = m.range(min=0.8;max=0.8)
                  );
                  relativeV = true;
                  robotMaxV = 0.1;
                  robotRadius = 0.01
                )
                """
        );

    @SuppressWarnings("unchecked") HebbianMultilayerPerceptron hmlp = ((Builder<HebbianMultilayerPerceptron, ?>) nb
        .build("ds.num.hebbianMlp(innerLayers = [16]; learningRate = 0.02)"))
        .apply(environment.exampleAgent().nOfInputs(), environment.exampleAgent().nOfOutputs());
    hmlp.randomize(new Random(2), DoubleRange.SYMMETRIC_UNIT);

    SingleAgentTask<NumericalDynamicalSystem<?>, double[], double[], NavigationEnvironment.State> task = SingleAgentTask
        .fromEnvironment(() -> environment, s -> false, true);
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>> outcome = task.simulate(
        hmlp,
        0.1,
        new DoubleRange(0, 30)
    );

    NavigationDrawer d = new NavigationDrawer(NavigationDrawer.Configuration.DEFAULT);
    @SuppressWarnings("unchecked") Function<Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>, Double> fitness = (Function<Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>, Double>) nb
        .build("ds.e.n.arenaCoverage()");
    System.out.println(fitness.apply(outcome));
    Function<Double, Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>> tResF = dT -> SingleAgentTask
        .fromEnvironment(
            () -> environment,
            s -> false,
            true
        )
        .simulate(hmlp, dT, new DoubleRange(0, 30));
    d.multi(Arrangement.HORIZONTAL)
        .show(
            DoubleStream.iterate(0.05, v -> v <= 0.25, v -> v + 0.025).boxed().map(tResF).toList()
        );
  }
}
