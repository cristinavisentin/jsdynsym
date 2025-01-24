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
  private PongFunctions() {}

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> score1(
      @Param(value = "of", dNPM = "f.identity()")
          Function<
                  X,
                  Simulation.Outcome<
                      HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>>
              beforeF,
      @Param(value = "format", dS = "%5.3f") String format) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f =
        o -> o.snapshots().lastEntry().getValue().state().lRacketState().score();
    return FormattedNamedFunction.from(f, format, "score.1").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> score2(
      @Param(value = "of", dNPM = "f.identity()")
          Function<
                  X,
                  Simulation.Outcome<
                      HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>>
              beforeF,
      @Param(value = "format", dS = "%5.3f") String format) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f =
        o -> o.snapshots().lastEntry().getValue().state().rRacketState().score();
    return FormattedNamedFunction.from(f, format, "score.2").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> scoreDiff1(
      @Param(value = "of", dNPM = "f.identity()")
          Function<
                  X,
                  Simulation.Outcome<
                      HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>>
              beforeF,
      @Param(value = "format", dS = "%5.3f") String format) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f =
        o -> o.snapshots().lastEntry().getValue().state().lRacketState().score()
            - o.snapshots()
                .lastEntry()
                .getValue()
                .state()
                .rRacketState()
                .score();
    return FormattedNamedFunction.from(f, format, "score.diff.1").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> scoreDiff2(
      @Param(value = "of", dNPM = "f.identity()")
          Function<
                  X,
                  Simulation.Outcome<
                      HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>>
              beforeF,
      @Param(value = "format", dS = "%5.3f") String format) {
    Function<Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double> f =
        o -> o.snapshots().lastEntry().getValue().state().rRacketState().score()
            - o.snapshots()
                .lastEntry()
                .getValue()
                .state()
                .lRacketState()
                .score();
    return FormattedNamedFunction.from(f, format, "score.diff.2").compose(beforeF);
  }
}
