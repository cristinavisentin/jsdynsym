package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.SimulationOutcomeDrawer;

import java.awt.*;
import java.util.function.Function;

public class PongDrawer
    implements SimulationOutcomeDrawer<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> {

  public record Configuration(
      Color racketsColor,
      Color ballColor,
      Color arenaColor,
      Color infoColor,
      double racketsThickness,
      double ballThickness,
      double arenaThickness,
      double racketsFillAlpha,
      double ballFillAlpha,
      double ballRadius,
      double marginRate
  ) {

    public static final Configuration DEFAULT = new Configuration(
        Color.BLUE,
        Color.MAGENTA,
        Color.DARK_GRAY,
        Color.BLUE,
        1,
        1,
        1,
        0.95,
        0.95,
        0.5,
        0.1
    );
  }

  @Override
  public void drawSingle(Graphics2D g, double t, HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State> stateStep) {
    PongEnvironment.State state = stateStep.state();
    PongDrawer.Configuration configuration = Configuration.DEFAULT;
    double arenaWidth = state.configuration().arenaXLength();
    double arenaHeight = state.configuration().arenaYLength();
    // compute margins to fit the arena to the design
    int panelWidth = g.getClipBounds().width;
    int panelHeight = g.getClipBounds().height;
    double margin = configuration.marginRate() * Math.min(panelWidth, panelHeight);
    // scale to fit game co-ordinates to drawing
    double scaleX = (panelWidth - 2 * margin) / arenaWidth;
    double scaleY = (panelHeight - 2 * margin) / arenaHeight;
    double scale = Math.min(scaleX, scaleY);
    // some useful functions for offsets
    double offsetX = (panelWidth - scale * arenaWidth) / 2.0;
    double offsetY = (panelHeight - scale * arenaHeight) / 2.0;
    Function<Double, Integer> screenX = x -> (int) (offsetX + x * scale);
    Function<Double, Integer> screenY = x -> (int) (offsetY + (arenaHeight - x) * scale);
    Function<Double, Integer> screenLength = x -> (int) (x * scale);
    // draw the arena
    g.setColor(configuration.arenaColor());
    g.setStroke(new BasicStroke((float) configuration.arenaThickness()));
    g.drawRect(screenX.apply(0.0), screenY.apply(arenaHeight), screenLength.apply(arenaWidth), screenLength.apply(arenaHeight));
    // draw the rackets
    g.setColor(configuration.racketsColor());
    for (PongEnvironment.RacketState racketState : new PongEnvironment.RacketState[]{
        state.lRacketState(), state.rRacketState()}) {
      double racketX = racketState.side() == PongEnvironment.Side.LEFT ? 0 : arenaWidth;
      double racketYCenter = racketState.yCenter();
      double edgeRadius = state.configuration().racketsEdgeRadius();
      double rectHeight = state.configuration().racketsLength() - 2 * edgeRadius;
      int alpha = (int) (configuration.racketsFillAlpha() * 255);
      g.setColor(new Color(configuration.racketsColor().getRed(), configuration.racketsColor().getGreen(), configuration.racketsColor().getBlue(), alpha));
      // draw the central rectangle
      g.fillRect(
          screenX.apply(racketX - edgeRadius),
          screenY.apply(racketYCenter + rectHeight / 2),
          screenLength.apply(2 * edgeRadius),
          screenLength.apply(rectHeight)
      );
      // draw the upper semicircle
      int arcWidth = screenLength.apply(2 * edgeRadius);
      int arcHeight = screenLength.apply(2 * edgeRadius);
      g.fillArc(
          screenX.apply(racketX - edgeRadius),
          screenY.apply(racketYCenter + rectHeight / 2 + edgeRadius),
          arcWidth,
          arcHeight,
          0,
          180
      );
      // draw the lower semicircle
      g.fillArc(
          screenX.apply(racketX - edgeRadius),
          screenY.apply(racketYCenter - rectHeight / 2 + edgeRadius),
          arcWidth,
          arcHeight,
          180,
          180
      );
    }
    // draw the ball
    PongEnvironment.BallState ballState = state.ballState();
    g.setColor(configuration.ballColor());
    int alpha = (int) (configuration.ballFillAlpha() * 255);
    g.setColor(new Color(configuration.ballColor().getRed(), configuration.ballColor().getGreen(), configuration.ballColor().getBlue(), alpha));
    double ballRadius = configuration.ballRadius();
    g.fillOval(
        screenX.apply(ballState.center().x() - ballRadius),
        screenY.apply(ballState.center().y() + ballRadius),
        screenLength.apply(ballRadius * 2),
        screenLength.apply(ballRadius * 2)
    );
    // draw the scores
    g.setColor(configuration.infoColor());
    g.setFont(new Font("Arial", Font.BOLD, 5));
    String scoreText = String.format("Left: %.0f  Right: %.0f", state.lRacketScore(), state.rRacketScore());
    g.drawString(scoreText, (int) margin * 2, (int) margin * 2);
  }
}
