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
import io.github.ericmedvet.jsdynsym.rl.Run.State;
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
  public static <X, C, O, A, S> NamedFunction<X, Outcome<Step<RewardedInput<O>, A, S>>> lastOutcome(
      @Param(value = "name", iS = "last.outcome") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, State<C, O, A, S>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<State<C, O, A, S>, Outcome<Step<RewardedInput<O>, A, S>>> f = State::lastOutcome;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, C, O, A, S> NamedFunction<X, C> agent(
      @Param(value = "name", iS = "agent") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, State<C, O, A, S>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<State<C, O, A, S>, C> f = State::agent;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, C, O, A, S> NamedFunction<X, Integer> nOfEpisodes(
      @Param(value = "name", iS = "n.episodes") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, State<C, O, A, S>> beforeF,
      @Param(value = "format", dS = "%5d") String format
  ) {
    Function<State<C, O, A, S>, Integer> f = State::nOfEpisodes;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> NamedFunction<X, Integer> nOfSteps(
      @Param(value = "name", iS = "n.steps") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, State<?, ?, ?, ?>> beforeF,
      @Param(value = "format", dS = "%7d") String format
  ) {
    Function<State<?, ?, ?, ?>, Integer> f = State::nOfSteps;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> elapsedSecs(
      @Param(value = "name", iS = "elapsed.secs") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, State<?, ?, ?, ?>> beforeF,
      @Param(value = "format", dS = "%6.1f") String format
  ) {
    Function<State<?, ?, ?, ?>, Double> f = s -> s.elapsedMillis() / 1000d;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

}
