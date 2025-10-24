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
package io.github.ericmedvet.jsdynsym.rl;

import io.github.ericmedvet.jgea.core.listener.Listener;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.core.ParamMap;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jsdynsym.control.Simulation.Outcome;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask.Step;
import io.github.ericmedvet.jsdynsym.control.SingleRLAgentTask;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent.RewardedInput;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

@Discoverable(prefixTemplate = "rl")
public record Run<C extends ReinforcementLearningAgent<O, A, AS>, O, A, AS, T extends SingleRLAgentTask<C, O, A, TS>, TS>(
    @Param(value = "", injection = Param.Injection.INDEX) int index,
    @Param(value = "name", dS = "") String name,
    @Param("agent") Function<? super C, ? extends C> agentSupplier,
    @Param("tasks") List<? extends T> tasks,
    @Param("dT") double dT,
    @Param("tRange") DoubleRange tRange,
    @Param("stopCriterion") Predicate<State<O, A, TS>> stopCriterion,
    @Param(value = "", injection = Param.Injection.MAP_WITH_DEFAULTS) ParamMap map
) {

  public record State<O, A, TS>(
      int nOfEpisodes,
      int nOfSteps,
      long elapsedMillis,
      Outcome<Step<RewardedInput<O>, A, TS>> lastOutcome
  ) {

  }

  public Outcome<Step<RewardedInput<O>, A, TS>> run(
      Listener<State<O, A, TS>> listener
  ) throws RunException {
    Instant startingT = Instant.now();
    C exampleAgent = tasks.getFirst()
        .example()
        .orElseThrow(() -> new RunException("Task has no example agent"));
    C agent = agentSupplier.apply(exampleAgent);
    State<O, A, TS> state = new State<>(0, 0, 0, null);
    try {
      while (Objects.isNull(state.lastOutcome) || !stopCriterion.test(state)) {
        T task = tasks.get(state.nOfEpisodes % tasks.size());
        Outcome<Step<RewardedInput<O>, A, TS>> outcome = task.simulate(agent, dT, tRange);
        state = new State<>(
            state.nOfEpisodes + 1,
            state.nOfSteps + outcome.snapshots().size(),
            Duration.between(startingT, Instant.now()).toMillis(),
            outcome
        );
        listener.listen(state);
      }
    } finally {
      listener.done();
    }
    return state.lastOutcome;
  }

}
