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

import io.github.ericmedvet.jgea.core.listener.ListenerFactory;
import io.github.ericmedvet.jgea.core.listener.TabularPrinter;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.FormattedFunction;
import io.github.ericmedvet.jsdynsym.rl.Experiment;
import io.github.ericmedvet.jsdynsym.rl.Run;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

@Discoverable(prefixTemplate = "rl.listener|l")
public class Listeners {

  private Listeners() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static BiFunction<Experiment, ExecutorService, ListenerFactory<? super Run.Iteration<?, ?, ?>, Run<?, ?, ?, ?, ?, ?>>> console(
      @Param(
          value = "defaultFunctions", dNPMs = {"rl.f.index()", "rl.f.cumulativeSteps()", "ds.f.cumulatedReward(of = rl.f.outcome())"}) List<Function<? super Run.Iteration<?, ?, ?>, ?>> defaultEpisodeFunctions,
      @Param(value = "functions") List<Function<? super Run.Iteration<?, ?, ?>, ?>> episodeFunctions,
      @Param(
          value = "defaultRunFunctions", dNPMs = {"rl.f.runKey(key = \"run.agent.name\")"}) List<Function<? super Run<?, ?, ?, ?, ?, ?>, ?>> defaultRunFunctions,
      @Param("runFunctions") List<Function<? super Run<?, ?, ?, ?, ?, ?>, ?>> runFunctions,
      @Param(value = "deferred") boolean deferred,
      @Param(value = "onlyLast", dB = true) boolean onlyLast,
      @Param(value = "runCondition", dNPM = "predicate.always()") Predicate<Run<?, ?, ?, ?, ?, ?>> runPredicate,
      @Param(value = "episodeCondition", dNPM = "predicate.always()") Predicate<Run.Iteration<?, ?, ?>> episodePredicate,
      @Param("logExceptions") boolean logExceptions

  ) {
    return (experiment, executor) -> {
      ListenerFactory<? super Run.Iteration<?, ?, ?>, Run<?, ?, ?, ?, ?, ?>> factory = new TabularPrinter<>(
          Stream.of(defaultEpisodeFunctions, episodeFunctions)
              .flatMap(List::stream)
              .toList(),
          Stream.concat(defaultRunFunctions.stream(), runFunctions.stream())
              .map(f -> reformatToFit(f, experiment.runs()))
              .toList(),
          episodePredicate,
          logExceptions
      );
      if (deferred) {
        factory = factory.deferred(executor);
      }
      if (onlyLast) {
        factory = factory.onLast();
      }
      return factory.conditional(runPredicate);
    };
  }

  private static <T, R> Function<T, R> reformatToFit(Function<T, R> f, Collection<?> ts) {
    //noinspection unchecked
    return FormattedFunction.from(f)
        .reformattedToFit(ts.stream().map(t -> (T) t).toList());
  }

}
