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
package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiEnvironment;
import io.github.ericmedvet.jsdynsym.control.geometry.Circle;
import io.github.ericmedvet.jsdynsym.control.geometry.Point;
import io.github.ericmedvet.jsdynsym.control.geometry.Rectangle;
import io.github.ericmedvet.jsdynsym.control.geometry.Segment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.random.RandomGenerator;
import java.util.stream.DoubleStream;

public class PongEnvironment implements HomogeneousBiEnvironment<double[], double[], PongEnvironment.State> {

  public record Configuration(
      DoubleRange racketsInitialYRange,
      double racketsLength,
      double racketsEdgeRadius,
      // the edges of the rackets are made by semi-circumferences, thus the rackets width is equal to
      // 2*racketsEdgeRadius
      double racketsMaxYVelocity,
      double racketsFriction,
      double spinEffectFactor,
      double ballInitialVelocity,
      double ballMaxVelocity,
      DoubleRange ballInitialAngleRange,
      double ballAccelerationRatePerPoint, // The ball accelerates during a point
      double arenaXLength,
      double arenaYLength,
      double maximumTimePerPoint,
      RandomGenerator randomGenerator) {
    public static final Configuration DEFAULT = new Configuration(
        new DoubleRange(22, 28),
        5,
        1.5,
        10,
        0.05,
        0.001,
        30,
        80,
        new DoubleRange(-Math.PI / 8, Math.PI / 8),
        1.05,
        60,
        50,
        Double.MAX_VALUE,
        new Random());
  }

  // All coordinate are in Arena reference frame that is centered in the down-left cornet, with the x-axis pointing
  // rightwards and the y-axis pointing upwards
  public record State(
      Configuration configuration,
      RacketState lRacketState,
      RacketState rRacketState,
      BallState ballState,
      double lRacketScore,
      double rRacketScore) {
  }

  private final Configuration configuration;
  private State state;
  private double previousTime;
  private PointStatus pointStatus;
  private double pointTime;

  public PongEnvironment(Configuration configuration) {
    this.configuration = configuration;
    reset();
  }

  public enum Side {
    LEFT,
    RIGHT
  }

  private enum PointStatus {
    IN_GAME,
    DRAW,
    L_WON,
    R_WON
  }

  public record RacketState(double yCenter, double yVelocity, Side side) {
    RacketState update(double yCenter, double yVelocity) {
      return new RacketState(yCenter, yVelocity, this.side);
    }

    RacketState deepCopy() {
      return new RacketState(yCenter, this.yVelocity, this.side);
    }
  }

  private double[] toNormalizedArray(RacketState rRacketState) {
    return new double[]{
        rRacketState.yCenter / configuration.arenaYLength,
        rRacketState.yVelocity / configuration.racketsMaxYVelocity
    };
  }

  private double[] toNormalizedArray(BallState ballState) {
    double normalizedBallXVelocity = ballState.velocity().x() / configuration.ballMaxVelocity;
    double normalizedBallYVelocity = ballState.velocity().y() / configuration.ballMaxVelocity;
    return new double[]{
        ballState.center().x() / configuration.arenaXLength,
        ballState.center().y() / configuration.arenaYLength,
        normalizedBallXVelocity,
        normalizedBallYVelocity
    };
  }

  public record BallState(Point center, Point velocity) {
    BallState rotateCounterClockWise(Point centerOfRotation, double angle) {
      Point ballCenterRotated = center().rotateCounterClockWise(
          centerOfRotation, angle);
      Point ballVelocityRotated =
          velocity().rotateCounterClockWise(centerOfRotation, angle);
      return new BallState(ballCenterRotated, ballVelocityRotated);
    }

    BallState deepCopy() {
      return new BallState(this.center(), this.velocity());
    }
  }

  public int nOfInputsPerAgent() {
    return 1;
  }

  @Override
  public Pair<double[], double[]> defaultActions() {
    return new Pair<>(new double[nOfInputsPerAgent()], new double[nOfInputsPerAgent()]);
  }

  @Override
  public State getState() {
    return state;
  }

  // TODO check
  private BallState handleCollisionWithRacketIfAny(BallState updatedBallState, RacketState racketState) {
    // each racket is composed by a rectangle in the center and two semicircles, one on the top and one on the
    // bottom of the rectangle
    double rectangleHeight = configuration.racketsLength - 2 * configuration.racketsEdgeRadius;
    Rectangle racketRectangle = new Rectangle(
        new Point(-configuration.racketsEdgeRadius, rectangleHeight / 2),
        new Point(configuration.racketsEdgeRadius, -rectangleHeight / 2));
    Circle upperCircle = new Circle(new Point(0, rectangleHeight / 2), configuration.racketsEdgeRadius);
    Circle lowerCircle = new Circle(new Point(0, -rectangleHeight / 2), configuration.racketsEdgeRadius);
    // get the trajectory made by the ball in the current time-step
    BallState previousBallStateRRF =
        toRacketReferenceFrame(state.ballState, racketState); // RRF -> Racket Reference Frame
    BallState updatedBallStateRRF = toRacketReferenceFrame(updatedBallState, racketState);
    Segment racketTrajectoryRRF = new Segment(previousBallStateRRF.center, updatedBallStateRRF.center);
    // check collisions with the racket components
    List<Point> upperCircleIntersections = upperCircle.intersection(racketTrajectoryRRF);
    List<Point> lowerCircleIntersections = lowerCircle.intersection(racketTrajectoryRRF);
    List<Point> rectangleVerticalEdgesIntersections =
        racketRectangle.verticalEdgesIntersections(racketTrajectoryRRF);
    List<Point> allRacketIntersections = new ArrayList<>();
    allRacketIntersections.addAll(upperCircleIntersections);
    allRacketIntersections.addAll(lowerCircleIntersections);
    allRacketIntersections.addAll(rectangleVerticalEdgesIntersections);
    // return if no collisions are detected
    if (allRacketIntersections.isEmpty()) {
      return updatedBallState;
    }
    // find the collision point that is closest to the ball initial position
    Point collisionPoint = allRacketIntersections.getFirst();
    allRacketIntersections.removeFirst();
    double minDistance = previousBallStateRRF.center.distance(collisionPoint);
    for (Point point : allRacketIntersections) {
      double distance = previousBallStateRRF.center.distance(point);
      if (distance < minDistance) {
        minDistance = distance;
        collisionPoint = point;
      }
    }
    // process the collision defending on the racket component
    if (rectangleVerticalEdgesIntersections.contains(collisionPoint)) {
      return toArenaReferenceFrame(
          racketVerticalEdgeCollisionWithSimplifiedSpinEffect(
              updatedBallStateRRF, racketState, collisionPoint),
          racketState);
    } else if (upperCircleIntersections.contains(collisionPoint)) {
      return toArenaReferenceFrame(
          racketCircularEdgeCollisionWithSimplifiedSpinEffect(
              updatedBallStateRRF, racketState, collisionPoint, upperCircle.center()),
          racketState);
    } else if (lowerCircleIntersections.contains(collisionPoint)) {
      return toArenaReferenceFrame(
          racketCircularEdgeCollisionWithSimplifiedSpinEffect(
              updatedBallStateRRF, racketState, collisionPoint, lowerCircle.center()),
          racketState);
    } else {
      throw new RuntimeException("Unhandled state: " + state.toString());
    }
  }

  private BallState racketVerticalEdgeCollisionWithSimplifiedSpinEffect(
      BallState updatedBallState, RacketState racketState, Point collisionPoint) {
    Point mirroredReflection = new Point(
        collisionPoint.x() + (collisionPoint.x() - updatedBallState.center.x()), updatedBallState.center.y());
    Point mirroredVelocity = new Point(
        -updatedBallState.velocity().x(), updatedBallState.velocity().y());
    double correctionAngle = -(configuration.spinEffectFactor * racketState.yVelocity); // The minus sign is needed to simulate inverse proportionality that arises with
    // the spin effect
    Point adjustedReflection = mirroredReflection.rotateCounterClockWise(collisionPoint, correctionAngle);
    Point adjustedVelocity = mirroredVelocity.rotateCounterClockWise(collisionPoint, correctionAngle);
    return new BallState(adjustedReflection, adjustedVelocity);
  }

  // TODO check
  private BallState racketCircularEdgeCollisionWithSimplifiedSpinEffect(
      BallState updatedBallState, RacketState racketState, Point collisionPoint, Point circularEdgeCenter) {
    double collisionPointRotationAngel = collisionPoint.getRotationAngleCounterClockwise(circularEdgeCenter);
    BallState ballStateRotatedVertically =
        updatedBallState.rotateCounterClockWise(circularEdgeCenter, -collisionPointRotationAngel);
    Point collisionPointRotatedVertically =
        collisionPoint.rotateCounterClockWise(circularEdgeCenter, -collisionPointRotationAngel);
    BallState ballStateAfterVerticalCollision = racketVerticalEdgeCollisionWithSimplifiedSpinEffect(
        ballStateRotatedVertically, racketState, collisionPointRotatedVertically);
    return ballStateAfterVerticalCollision.rotateCounterClockWise(circularEdgeCenter, collisionPointRotationAngel);
  }

  // The racket reference frame is centered in the racket center with the x-axis pointing the center of the arena and
  // the y-axis pointing upwards
  private BallState toRacketReferenceFrame(BallState ballState, RacketState racketState) {
    BallState ballStateFlippedReferenceFrame = ballState.deepCopy();
    if (racketState.side.equals(Side.RIGHT)) {
      ballStateFlippedReferenceFrame = flippedHorizontalAxisReferenceFrame(ballStateFlippedReferenceFrame);
    }
    return new BallState(
        new Point(
            ballStateFlippedReferenceFrame.center.x(),
            ballStateFlippedReferenceFrame.center.y() - racketState.yCenter
        ),
        ballStateFlippedReferenceFrame.velocity);
  }

  private BallState toArenaReferenceFrame(BallState ballState, RacketState racketState) {
    BallState ballStateFlippedReferenceFrame = ballState.deepCopy();
    if (racketState.side.equals(Side.RIGHT)) {
      ballStateFlippedReferenceFrame = flippedHorizontalAxisReferenceFrame(ballStateFlippedReferenceFrame);
    }
    return new BallState(
        new Point(
            ballStateFlippedReferenceFrame.center.x(),
            ballStateFlippedReferenceFrame.center.y() + racketState.yCenter
        ),
        ballStateFlippedReferenceFrame.velocity);
  }

  private BallState flippedHorizontalAxisReferenceFrame(BallState ballState) {
    return new BallState(
        new Point(configuration.arenaXLength() - ballState.center.x(), ballState.center.y()),
        new Point(-ballState.velocity.x(), ballState.velocity.y()));
  }

  private void checkPointEnd(BallState updatedBallState) {
    if (updatedBallState.center.x() < 0) {
      pointStatus = PointStatus.R_WON;
    } else if (updatedBallState.center.x() > configuration.arenaXLength()) {
      pointStatus = PointStatus.L_WON;
    } else if (pointTime >= configuration.maximumTimePerPoint) {
      pointStatus = PointStatus.DRAW;
    }
    /*
    if (pointStatus == PointStatus.IN_GAME) {
    Segment trajectory = new Segment(state.ballState.center, updatedBallState.center);
    List<Point> pointEndingIntersections = arena.verticalEdgesIntersections(trajectory);
    if (!pointEndingIntersections.isEmpty()) {
    if (pointEndingIntersections.getFirst().x() < configuration.arenaXLength / 2.0) {
    pointStatus = PointStatus.IN_GAME;
    } else {
    pointStatus = PointStatus.L_WON;
    }
    } else {
    if (t >= configuration.maximumTimePerPoint) {
    pointStatus = PointStatus.DRAW;
    }
    }
    }
    */
  }

  private BallState handleArenaEdgeMirroredVerticalCollisionIfAny(BallState ballState) {
    double offsetAbove = configuration.arenaYLength - ballState.center.y();
    double offsetBelow = ballState.center.y();
    if (offsetBelow * offsetAbove > 0.0d) {
      return ballState;
    }
    Point updatedBallVelocity =
        new Point(ballState.velocity().x(), -ballState.velocity().y());
    Point updatedBallPosition;
    if (offsetAbove < offsetBelow) {
      updatedBallPosition = new Point(ballState.center.x(), configuration.arenaYLength + offsetAbove);
    } else {
      updatedBallPosition = new Point(ballState.center.x(), -offsetBelow);
    }
    return new BallState(updatedBallPosition, updatedBallVelocity);
  }

  private BallState handleCollisionWithRacketsIfAny(BallState updatedBallState) {
    BallState processedBallStateCollisionLeftRacket =
        handleCollisionWithRacketIfAny(updatedBallState, state.lRacketState);
    if (processedBallStateCollisionLeftRacket.equals(updatedBallState)) {
      return handleCollisionWithRacketIfAny(updatedBallState, state.rRacketState);
    } else {
      return processedBallStateCollisionLeftRacket;
    }
  }

  private Point randomBallVelocity() {
    double ballInitialAngle =
        configuration.ballInitialAngleRange.denormalize(configuration.randomGenerator.nextDouble());
    boolean flipBallVelocity = configuration.randomGenerator.nextBoolean();
    if (flipBallVelocity) {
      ballInitialAngle = ballInitialAngle + Math.PI;
    }
    double updatedBallInitialVelocity;
    try {
      updatedBallInitialVelocity = configuration.ballInitialVelocity
          * Math.pow(configuration.ballAccelerationRatePerPoint, (state.lRacketScore + state.rRacketScore));
      if (updatedBallInitialVelocity > configuration.ballMaxVelocity){
        updatedBallInitialVelocity = configuration.ballMaxVelocity;
      }
    } catch (NullPointerException e) {
      updatedBallInitialVelocity = configuration.ballInitialVelocity;
    }
    double ballInitialXV = Math.cos(ballInitialAngle) * updatedBallInitialVelocity;
    double ballInitialYV = Math.sin(ballInitialAngle) * updatedBallInitialVelocity;
    return new Point(ballInitialXV, ballInitialYV);
  }

  @Override
  public void reset() {
    state = new State(
        configuration,
        new RacketState(
            configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
            0,
            Side.LEFT),
        new RacketState(
            configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
            0,
            Side.RIGHT),
        new BallState(
            new Point(configuration.arenaXLength / 2.0, configuration.arenaYLength / 2.0),
            randomBallVelocity()),
        0,
        0);
    previousTime = 0.0;
    pointTime = 0.0;
    pointStatus = PointStatus.IN_GAME;
  }

  private RacketState updateRacketState(RacketState racketState, double normalizedAction, double deltaTime) {
    double updatedRacketYVelocity = racketState.yVelocity
        + normalizedAction * configuration.racketsMaxYVelocity
        - deltaTime * configuration.racketsFriction * racketState.yVelocity;
    if (updatedRacketYVelocity > configuration.racketsMaxYVelocity) {
      updatedRacketYVelocity = configuration.racketsMaxYVelocity;
    } else if (updatedRacketYVelocity < -configuration.racketsMaxYVelocity) {
      updatedRacketYVelocity = -configuration.racketsMaxYVelocity;
    }
    double updatedRacketY = racketState.yCenter + updatedRacketYVelocity * deltaTime;
    // RacketState updatedRacketState = racketState.update(updatedRacketY, updatedRacketYVelocity);
    double upperEdgeArenaOffset = configuration.arenaYLength - (updatedRacketY + configuration.racketsLength / 2);
    double lowerEdgeArenaOffset = updatedRacketY - configuration.racketsLength / 2;
    if (!(upperEdgeArenaOffset * lowerEdgeArenaOffset > 0.0)) {
      updatedRacketYVelocity = -updatedRacketYVelocity;
      if (upperEdgeArenaOffset < lowerEdgeArenaOffset) {
        updatedRacketY = updatedRacketY + upperEdgeArenaOffset;
      } else {
        updatedRacketY = updatedRacketY - lowerEdgeArenaOffset;
      }
    }
    return racketState.update(updatedRacketY, updatedRacketYVelocity);
  }

  private BallState updateBallState(BallState ballState, double deltaTime) {
    double deltaX = ballState.velocity.x() * deltaTime;
    double deltaY = ballState.velocity.y() * deltaTime;
    Point deltaPosition = new Point(deltaX, deltaY);
    return new BallState(ballState.center.sum(deltaPosition), ballState.velocity);
  }

  private void updateState(BallState ballState, RacketState lRacketState, RacketState rRacketState) {
    if (pointStatus != PointStatus.IN_GAME) {
      double lRacketScore = state.lRacketScore;
      double rRacketScore = state.rRacketScore;
      switch (pointStatus) {
        case DRAW -> {
          lRacketScore = lRacketScore + 0.5;
          rRacketScore = rRacketScore + 0.5;
        }
        case L_WON -> lRacketScore = lRacketScore + 1;
        case R_WON -> rRacketScore = rRacketScore + 1;
      }
      state = new State(
          configuration,
          new RacketState(
              configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
              0,
              Side.LEFT),
          new RacketState(
              configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
              0,
              Side.RIGHT),
          new BallState(
              new Point(configuration.arenaXLength / 2.0, configuration.arenaYLength / 2.0),
              randomBallVelocity()),
          lRacketScore,
          rRacketScore);
      pointTime = 0d;
      pointStatus = PointStatus.IN_GAME;
    } else {
      state = new State(
          configuration, lRacketState, rRacketState, ballState, state.lRacketScore, state.rRacketScore);
    }
  }

  @Override
  public Pair<double[], double[]> step(double t, Pair<double[], double[]> normalizedActions) {
    if (normalizedActions.first().length != nOfInputsPerAgent()) {
      throw new IllegalArgumentException("Left agent action has wrong number of elements: %d found, %d expected"
          .formatted(normalizedActions.first().length, nOfInputsPerAgent()));
    }
    if (normalizedActions.second().length != nOfInputsPerAgent()) {
      throw new IllegalArgumentException("Right agent action has wrong number of elements: %d found, %d expected"
          .formatted(normalizedActions.second().length, nOfInputsPerAgent()));
    }
    // prepare
    double deltaTime = t - previousTime;
    previousTime = t;
    RacketState updatedLRacketState =
        updateRacketState(state.lRacketState, normalizedActions.first()[0], deltaTime);
    RacketState updatedRRacketState =
        updateRacketState(state.rRacketState, normalizedActions.second()[0], deltaTime);
    BallState updatedBallState = updateBallState(state.ballState, deltaTime);
    BallState updatedBallStateWithCollision = updatedBallState.deepCopy();
    updatedBallStateWithCollision = handleArenaEdgeMirroredVerticalCollisionIfAny(updatedBallStateWithCollision);
    updatedBallStateWithCollision = handleCollisionWithRacketsIfAny(updatedBallStateWithCollision);
    updatedBallStateWithCollision = handleArenaEdgeMirroredVerticalCollisionIfAny(updatedBallStateWithCollision);
    updatedBallStateWithCollision = handleCollisionWithRacketsIfAny(updatedBallStateWithCollision);

    pointTime = pointTime + deltaTime;
    checkPointEnd(updatedBallStateWithCollision);

    updateState(updatedBallStateWithCollision, updatedLRacketState, updatedRRacketState);
    double[] lRacketObservation = DoubleStream.concat(
            Arrays.stream(toNormalizedArray(updatedLRacketState)),
            Arrays.stream(toNormalizedArray(updatedBallStateWithCollision)))
        .toArray();
    double[] rRacketObservation = DoubleStream.concat(
            Arrays.stream(toNormalizedArray(updatedRRacketState)),
            Arrays.stream(
                toNormalizedArray(flippedHorizontalAxisReferenceFrame(updatedBallStateWithCollision))))
        .toArray();
    return new Pair<>(lRacketObservation, rRacketObservation);
  }
}
