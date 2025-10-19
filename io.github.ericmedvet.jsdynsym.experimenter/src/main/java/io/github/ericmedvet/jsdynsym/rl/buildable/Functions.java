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
package io.github.ericmedvet.jsdynsym.rl.buildable;

import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jnb.datastructure.NamedFunction;
import io.github.ericmedvet.jsdynsym.control.Simulation.Outcome;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask.Step;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent.RewardedInput;
import io.github.ericmedvet.jsdynsym.rl.Run;
import io.github.ericmedvet.jsdynsym.rl.Run.Iteration;
import io.github.ericmedvet.jsdynsym.rl.Utils;
import java.util.function.Function;

@Discoverable(prefixTemplate = "rl.function|f")
public class Functions {

  private Functions() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, String> runKey(
      @Param(value = "name", iS = "{key}") String name,
      @Param("key") String key,
      @Param(value = "of", dNPM = "f.identity()") Function<X, Run<?, ?, ?, ?, ?, ?>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<Run<?, ?, ?, ?, ?, ?>, String> f = run -> Utils.interpolate(
        "{%s}".formatted(key),
        null,
        run
    );
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, String> runString(
      @Param(value = "name", iS = "{s}") String name,
      @Param("s") String s,
      @Param(value = "of", dNPM = "f.identity()") Function<X, Run<?, ?, ?, ?, ?, ?>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<Run<?, ?, ?, ?, ?, ?>, String> f = run -> Utils.interpolate(s, null, run);
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, O, A, S> NamedFunction<X, Outcome<Step<RewardedInput<O>, A, S>>> outcome(
      @Param(value = "name", iS = "outcome") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, Run.Iteration<O, A, S>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<Run.Iteration<O, A, S>, Outcome<Step<RewardedInput<O>, A, S>>> f = Iteration::outcome;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, O, A, S> NamedFunction<X, Integer> index(
      @Param(value = "name", iS = "index") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, Run.Iteration<O, A, S>> beforeF,
      @Param(value = "format", dS = "%4d") String format
  ) {
    Function<Run.Iteration<O, A, S>, Integer> f = Iteration::index;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, O, A, S> NamedFunction<X, Integer> cumulativeSteps(
      @Param(value = "name", iS = "cumulative.steps") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, Run.Iteration<O, A, S>> beforeF,
      @Param(value = "format", dS = "%6d") String format
  ) {
    Function<Run.Iteration<O, A, S>, Integer> f = Iteration::cumulativeSteps;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }
}
