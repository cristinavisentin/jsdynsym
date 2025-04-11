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
package io.github.ericmedvet.jsdynsym.buildable.builders;

import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment;
import java.util.function.Function;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.environment|env|e.pong")
public class PongFunctions {
  private PongFunctions() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> numberOfCollisionsWithBall1(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.0f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> (double) o
        .snapshots()
        .lastEntry()
        .getValue()
        .state()
        .lRacketState()
        .nOfBallCollisions();
    return FormattedNamedFunction.from(f, format, "number.of.collisions.with.ball.1")
        .compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> numberOfCollisionsWithBall2(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.0f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> (double) o
        .snapshots()
        .lastEntry()
        .getValue()
        .state()
        .rRacketState()
        .nOfBallCollisions();
    return FormattedNamedFunction.from(f, format, "number.of.collisions.with.ball.2")
        .compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> yOffsetFromBall1(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.0f") String format,
      @Param(value = "ballXProximityThreshold", dD = 0.2) double ballXProximityThreshold
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> {
      double distanceThreshold = o.snapshots().lastEntry().getValue().state().configuration().arenaXLength() * ballXProximityThreshold;
      return o.snapshots().values().stream()
          .map(HomogeneousBiAgentTask.Step::state)
          .filter(s -> s.ballState().position().x() < distanceThreshold)
          .mapToDouble(s -> Math.abs(s.lRacketState().yCenter() - s.ballState().position().y()))
          .average()
          .orElse(distanceThreshold); //TODO check if this is correct
    };
    return FormattedNamedFunction.from(f, format, "y.offset.from.ball.1")
        .compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> yOffsetFromBall2(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.0f") String format,
      @Param(value = "ballXProximityThreshold", dD = 0.2) double ballXProximityThreshold
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> {
      double distanceThreshold = o.snapshots().lastEntry().getValue().state().configuration().arenaXLength() * (1 - ballXProximityThreshold);
      return o.snapshots().values().stream()
          .map(HomogeneousBiAgentTask.Step::state)
          .filter(s -> s.ballState().position().x() > distanceThreshold)
          .mapToDouble(s -> Math.abs(s.rRacketState().yCenter() - s.ballState().position().y()))
          .average()
          .orElse(distanceThreshold); //TODO check if this is correct
    };
    return FormattedNamedFunction.from(f, format, "y.offset.from.ball.2")
        .compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> score1(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.3f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> o
        .snapshots()
        .lastEntry()
        .getValue()
        .state()
        .lRacketState()
        .score();
    return FormattedNamedFunction.from(f, format, "score.1").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> score2(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.3f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> o
        .snapshots()
        .lastEntry()
        .getValue()
        .state()
        .rRacketState()
        .score();
    return FormattedNamedFunction.from(f, format, "score.2").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> scoreDiff1(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.3f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> {
      double s1 = o.snapshots().lastEntry().getValue().state().lRacketState().score();
      double s2 = o.snapshots().lastEntry().getValue().state().rRacketState().score();
      return s1 - s2;
    };
    return FormattedNamedFunction.from(f, format, "score.diff.1").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> scoreDiff2(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.3f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> {
      double s1 = o.snapshots().lastEntry().getValue().state().lRacketState().score();
      double s2 = o.snapshots().lastEntry().getValue().state().rRacketState().score();
      return s2 - s1;
    };
    return FormattedNamedFunction.from(f, format, "score.diff.2").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> shiftedScoreDiff1(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.3f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> {
      double s1 = o.snapshots().lastEntry().getValue().state().lRacketState().score();
      double s2 = o.snapshots().lastEntry().getValue().state().rRacketState().score();
      return s1 - s2 - Math.min(s1, s2);
    };
    return FormattedNamedFunction.from(f, format, "shifted.score.diff.1").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> shiftedScoreDiff2(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>> beforeF,
      @Param(value = "format", dS = "%5.3f") String format
  ) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f = o -> {
      double s1 = o.snapshots().lastEntry().getValue().state().lRacketState().score();
      double s2 = o.snapshots().lastEntry().getValue().state().rRacketState().score();
      return s2 - s1 - Math.min(s1, s2);
    };
    return FormattedNamedFunction.from(f, format, "shifted.score.diff.2").compose(beforeF);
  }
}
