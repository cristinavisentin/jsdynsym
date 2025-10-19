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
import io.github.ericmedvet.jgea.core.listener.ListenerFactory;
import io.github.ericmedvet.jnb.core.ProjectInfoProvider;
import io.github.ericmedvet.jsdynsym.control.Simulation.Outcome;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask.Step;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent.RewardedInput;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class Experimenter {

  private static final Logger L = Logger.getLogger(Experimenter.class.getName());

  private final ExecutorService runExecutorService;
  private final ExecutorService listenerExecutorService;

  public Experimenter(int nOfConcurrentRuns) {
    this.runExecutorService = Executors.newFixedThreadPool(nOfConcurrentRuns);
    this.listenerExecutorService = Executors.newVirtualThreadPerTaskExecutor();
  }

  public void run(Experiment experiment, boolean verbose) {
    ProjectInfoProvider.of(getClass()).ifPresent(pi -> L.info("Starting %s".formatted(pi)));
    record RunOutcome(Run<?, ?, ?, ?, ?, ?> run, Future<List<Outcome<Step<RewardedInput<?>, ?, ?>>>> future) {}
    // prepare listeners
    @SuppressWarnings("unchecked") List<? extends ListenerFactory<Run.Iteration<?, ?, ?>, Run<?, ?, ?, ?, ?, ?>>> factories = experiment
        .listeners()
        .stream()
        .map(
            builder -> (ListenerFactory<Run.Iteration<?, ?, ?>, Run<?, ?, ?, ?, ?, ?>>) builder.apply(
                experiment,
                listenerExecutorService
            )
        )
        .toList();
    ListenerFactory<Run.Iteration<?, ?, ?>, Run<?, ?, ?, ?, ?, ?>> factory = ListenerFactory.all(
        factories
    );
    // submit jobs
    List<RunOutcome> runOutcomes = experiment.runs()
        .stream()
        .map(run -> new RunOutcome(run, runExecutorService.submit(() -> {
          L.fine("Starting run %d of %d ".formatted(run.index() + 1, experiment.runs().size()));
          Instant startingT = Instant.now();
          @SuppressWarnings({"unchecked", "rawtypes"}) List<Outcome<Step<RewardedInput<?>, ?, ?>>> outcomes = run.run(
              (Listener) factory.build(run)
          );
          double elapsedT = Duration.between(startingT, Instant.now()).toMillis() / 1000d;
          L.fine(
              String.format(
                  "Run %d of %d done in %.2fs with %d episodes",
                  run.index() + 1,
                  experiment.runs().size(),
                  elapsedT,
                  outcomes.size()
              )
          );
          return outcomes;
        })))
        .toList();
    // wait for results
    runOutcomes.forEach(runOutcome -> {
      try {
        runOutcome.future.get();
      } catch (InterruptedException | ExecutionException e) {
        L.warning(String.format("Cannot solve %s: %s", runOutcome.run().map(), e));
        if (verbose) {
          //noinspection CallToPrintStackTrace
          e.printStackTrace();
        }
      }
    });
    // close
    L.info("Closing");
    runExecutorService.shutdown();
    listenerExecutorService.shutdown();
    while (true) {
      try {
        if (listenerExecutorService.awaitTermination(1, TimeUnit.SECONDS)) {
          break;
        }
      } catch (InterruptedException e) {
        // ignore
      }
    }
    try {
      factory.shutdown();
    } catch (Throwable e) {
      L.warning(String.format("Listener %s cannot shutdown() event: %s", factory, e));
      if (verbose) {
        //noinspection CallToPrintStackTrace
        e.printStackTrace();
      }
    }
  }
}
