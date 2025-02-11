/*-
 * ========================LICENSE_START=================================
 * jsdynsym-control
 * %%
 * Copyright (C) 2023 - 2024 Eric Medvet
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

package io.github.ericmedvet.jsdynsym.control.geometry;

import java.util.stream.DoubleStream;

public record Point(double x, double y) {

  public static Point ORIGIN = new Point(0, 0);

  public Point(double direction) {
    this(Math.cos(direction), Math.sin(direction));
  }

  public Point diff(Point p) {
    return new Point(x - p.x(), y - p.y());
  }

  public double direction() {
    return Math.atan2(y, x);
  }

  public double distance(Point p) {
    return diff(p).magnitude();
  }

  public double distance(Segment s) {
    return DoubleStream.of(
        Line.from(this, s.direction() + Math.PI / 2d)
            .interception(s)
            .map(p -> p.distance(this))
            .orElse(Double.POSITIVE_INFINITY),
        distance(s.p1()),
        distance(s.p2())
    )
        .min()
        .orElseThrow();
  }

  public double distance(Line l) {
    return Math.abs(l.a() * x + l.b() * y + l.c()) / Math.sqrt(l.a() * l.a() + l.b() * l.b());
  }

  public double magnitude() {
    return Math.sqrt(x * x + y * y);
  }

  public Point getOpposite() {
    return new Point(-x, -y);
  }

  public Point scale(double r) {
    return new Point(r * x, r * y);
  }

  public Point sum(Point p) {
    return new Point(x + p.x(), y + p.y());
  }

  // TODO check
  public Point rotate(Point centerOfRotation, double angle) {
    return this.translate(centerOfRotation.getOpposite()).rotate(angle).translate(centerOfRotation);
  }

  public Point translate(Point translation) {
    return new Point(x + translation.x(), y + translation.y());
  }

  public Point rotate(double angle) {
    return new Point(x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle));
  }

  // TODO check
  public double getRotationAngle(Point centerOfRotation) {
    return this.diff(centerOfRotation).direction();
  }

  @Override
  public String toString() {
    return String.format("(%.3f;%.3f)", x, y);
  }
}
