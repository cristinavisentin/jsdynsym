/*-
 * ========================LICENSE_START=================================
 * jsdynsym-control
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
package io.github.ericmedvet.jsdynsym.control.navigation;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jviz.core.drawer.Drawer;
import io.github.ericmedvet.jviz.core.util.GraphicsUtils;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;

public class ArenaDrawer implements Drawer<Arena> {

  private static final int DEFAULT_SIDE_LENGTH = 400;

  private final Configuration configuration;

  public record Configuration(
      Color startColor,
      Color targetColor,
      Color segmentColor,
      double segmentThickness,
      double landmarkAlpha,
      double landmarkMinSize,
      double marginRate
  ) {
    public static Configuration DEFAULT = new Configuration(
        Color.RED.darker(),
        Color.GREEN.darker(),
        Color.BLACK,
        3,
        0.5,
        10,
        0.01
    );
  }

  public ArenaDrawer(Configuration configuration) {
    this.configuration = configuration;
  }

  @Override
  public void draw(Graphics2D g, Arena arena) {
    // set transform
    AffineTransform previousTransform = setTransform(g, arena);
    // draw arena
    g.setStroke(
        new BasicStroke(
            (float) (configuration.segmentThickness / g.getTransform().getScaleX())
        )
    );
    g.setColor(configuration.segmentColor);
    arena.segments().forEach(s -> g.draw(new Line2D.Double(s.p1().x(), s.p1().y(), s.p2().x(), s.p2().y())));
    if (arena instanceof NavigationArena navigationArena) {
      drawLandmark(
          g,
          configuration.targetColor,
          configuration.landmarkMinSize / g.getTransform().getScaleX(),
          navigationArena.targetXRange(),
          navigationArena.targetYRange()
      );
      drawLandmark(
          g,
          configuration.startColor,
          configuration.landmarkMinSize / g.getTransform().getScaleX(),
          navigationArena.startXRange(),
          navigationArena.startYRange()
      );
    }
    g.setTransform(previousTransform);
  }

  protected AffineTransform setTransform(Graphics2D g, Arena arena) {
    double cX = g.getClipBounds().getX();
    double cY = g.getClipBounds().getY();
    double cW = g.getClipBounds().getWidth();
    double cH = g.getClipBounds().getHeight();
    // compute transformation
    double scale = Math.min(
        cW / (1d + 2d * configuration.marginRate) / arena.xExtent(),
        cH / (1d + 2d * configuration.marginRate) / arena.yExtent()
    );
    AffineTransform previousTransform = g.getTransform();
    AffineTransform transform = (AffineTransform) previousTransform.clone();
    transform.scale(scale, scale);
    transform.translate(
        (cX / scale + cW / scale - arena.xExtent()) / 2d,
        (cY / scale + cH / scale - arena.yExtent()) / 2d
    );
    g.setTransform(transform);
    return previousTransform;
  }

  private void drawLandmark(Graphics2D g, Color c, double minSize, DoubleRange xRange, DoubleRange yRange) {
    g.setColor(GraphicsUtils.alphaed(c, configuration.landmarkAlpha));
    if (xRange.extent() == 0) {
      xRange = new DoubleRange(
          xRange.min() - minSize / 2d,
          xRange.max() + minSize / 2d
      );
    }
    if (xRange.extent() < minSize) {
      xRange = xRange.extend(minSize / xRange.extent());
    }
    if (yRange.extent() == 0) {
      yRange = new DoubleRange(
          yRange.min() - minSize / 2d,
          yRange.max() + minSize / 2d
      );
    }
    if (yRange.extent() < minSize) {
      yRange = yRange.extend(minSize / yRange.extent());
    }
    g.fill(new Rectangle2D.Double(xRange.min(), yRange.min(), xRange.extent(), yRange.extent()));
  }

  @Override
  public ImageInfo imageInfo(Arena arena) {
    return new ImageInfo(
        (int) (arena.xExtent() > arena.yExtent() ? DEFAULT_SIDE_LENGTH * arena.xExtent() / arena
            .yExtent() : DEFAULT_SIDE_LENGTH),
        (int) (arena.xExtent() > arena.yExtent() ? DEFAULT_SIDE_LENGTH : DEFAULT_SIDE_LENGTH * arena.yExtent() / arena
            .xExtent())
    );
  }

}
