/*-
 * ========================LICENSE_START=================================
 * jsdynsym-core
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

package io.github.ericmedvet.jsdynsym.core.numerical.named;

import io.github.ericmedvet.jsdynsym.core.numerical.UnivariateRealFunction;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public interface NamedUnivariateRealFunction extends NamedMultivariateRealFunction, UnivariateRealFunction {
  double computeAsDouble(Map<String, Double> input);

  String yVarName();

  static <I> NamedUnivariateRealFunction from(
      ToDoubleFunction<Map<String, Double>> f,
      List<String> xVarNames,
      String yVarName,
      I inner
  ) {
    record HardComposedNURF<I>(
        ToDoubleFunction<Map<String, Double>> f,
        List<String> xVarNames,
        String yVarName,
        I inner
    ) implements io.github.ericmedvet.jsdynsym.core.composed.Composed<I>, NamedUnivariateRealFunction {

      @Override
      public double computeAsDouble(Map<String, Double> input) {
        return f.applyAsDouble(input);
      }

      @Override
      public String toString() {
        return Objects.isNull(inner) ? f.toString() : inner.toString();
      }
    }
    return new HardComposedNURF<>(f, xVarNames, yVarName, inner);
  }

  static NamedUnivariateRealFunction from(NamedMultivariateRealFunction nmrf) {
    return from(
        input -> nmrf.compute(input).get(nmrf.yVarNames().getFirst()),
        nmrf.xVarNames(),
        nmrf.yVarNames().getFirst(),
        nmrf
    );
  }

  static NamedUnivariateRealFunction from(UnivariateRealFunction urf, List<String> xVarNames, String yVarName) {
    return from(
        input -> {
          double[] in = xVarNames.stream().mapToDouble(input::get).toArray();
          if (in.length != urf.nOfInputs()) {
            throw new IllegalArgumentException(
                "Wrong input size: %d expected, %d found".formatted(urf.nOfInputs(), in.length)
            );
          }
          return urf.applyAsDouble(in);
        },
        xVarNames,
        yVarName,
        urf
    );
  }

  @Override
  default double applyAsDouble(double[] input) {
    return compute(input)[0];
  }

  @Override
  default Map<String, Double> compute(Map<String, Double> input) {
    return Map.of(yVarName(), computeAsDouble(input));
  }

  @Override
  default List<String> yVarNames() {
    return List.of(yVarName());
  }

  @Override
  default double[] compute(double... xs) {
    if (xs.length != xVarNames().size()) {
      throw new IllegalArgumentException(
          "Wrong number of inputs: %d expected, %d found"
              .formatted(xVarNames().size(), xs.length)
      );
    }
    Map<String, Double> output = compute(
        IntStream.range(0, xVarNames().size())
            .mapToObj(i -> Map.entry(xVarNames().get(i), xs[i]))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue))
    );
    return yVarNames().stream().mapToDouble(output::get).toArray();
  }

  @Override
  default int nOfOutputs() {
    return 1;
  }

  @Override
  default NamedUnivariateRealFunction andThen(DoubleUnaryOperator f) {
    NamedUnivariateRealFunction thisNurf = this;
    return NamedUnivariateRealFunction.from(
        UnivariateRealFunction.from(
            new ToDoubleFunction<>() {
              @Override
              public double applyAsDouble(double[] vs) {
                return f.applyAsDouble(thisNurf.applyAsDouble(vs));
              }

              @Override
              public String toString() {
                return thisNurf + "[then:%s]".formatted(f);
              }
            },
            nOfInputs()
        ),
        xVarNames(),
        yVarName()
    );
  }
}
