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
import java.util.ArrayList;
import java.util.List;
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
    @Param("stopCriterion") Predicate<List<Outcome<Step<RewardedInput<O>, A, TS>>>> stopCriterion,
    @Param(value = "", injection = Param.Injection.MAP_WITH_DEFAULTS) ParamMap map
) {

  public record Iteration<O, A, TS>(
      int index,
      int cumulativeSteps,
      Outcome<Step<RewardedInput<O>, A, TS>> outcome
  ) {}

  public List<Outcome<Step<RewardedInput<O>, A, TS>>> run(
      Listener<Iteration<O, A, TS>> listener
  ) throws RunException {
    List<Outcome<Step<RewardedInput<O>, A, TS>>> outcomes = new ArrayList<>();
    C exampleAgent = tasks.getFirst()
        .example()
        .orElseThrow(() -> new RunException("Task has no example agent"));
    C agent = agentSupplier.apply(exampleAgent);
    int indexOfTask = 0;
    int cumulativeSteps = 0;
    try {
      while (!stopCriterion.test(outcomes)) {
        T task = tasks.get(indexOfTask % tasks.size());
        Outcome<Step<RewardedInput<O>, A, TS>> outcome = task.simulate(agent, dT, tRange);
        cumulativeSteps = cumulativeSteps + outcome.snapshots().size();
        listener.listen(
            new Iteration<>(
                indexOfTask,
                cumulativeSteps,
                outcome
            )
        );
        outcomes.add(outcome);
        indexOfTask = indexOfTask + 1;
      }
    } finally {
      listener.done();
    }
    return outcomes;
  }

}
