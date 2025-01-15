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
      // the edges of the rackets are made by semi-circumferences, thus the rackets width is equal to 2*racketsEdgeRadius
      double racketsMaxYVelocity,
      double racketsFriction,
      double spinEffectFactor,
      DoubleRange ballInitialXVelocityRange,
      DoubleRange ballInitialYVelocityRange,
      double ballAccelerationRatePerPoint, // The ball accelerates during a point
      double arenaXLength,
      double arenaYLength,
      double maximumTimePerPoint,
      RandomGenerator randomGenerator
  ) {
    public static final Configuration DEFAULT = new Configuration(
        new DoubleRange(20, 30),
        10,
        1,
        5,
        0.5,
        0.1,
        new DoubleRange(2,5),
        new DoubleRange(2,5),
        1.5,
        100,
        50,
        Double.MAX_VALUE,
        new Random()
    );
  }

  // All coordinate are in Arena reference frame that is centered in the down-left cornet, with the x-axis pointing rightwards and the y-axis pointing upwards
  public record State(
      Configuration configuration,
      RacketState lRacketState,
      RacketState rRacketState,
      BallState ballState,
      double lRacketScore,
      double rRacketScore
  ) {
  }

  private final Configuration configuration;
  private State state;
  private double previousTime;
  private PointStatus pointStatus;
  private Rectangle arena;

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

    double[] toArray() {
      return new double[]{yCenter, yVelocity};
    }
  }

  public record BallState(Point center, Point velocity) {
    BallState rotateCounterClockWise(Point centerOfRotation, double angle) {
      Point ballCenterRotated = center().rotateCounterClockWise(centerOfRotation, -angle);
      Point ballVelocityRotated = velocity().rotateCounterClockWise(centerOfRotation, -angle);
      return new BallState(ballCenterRotated, ballVelocityRotated);
    }

    BallState deepCopy() {
      return new BallState(this.center(), this.velocity());
    }

    double[] toArray() {
      return new double[]{center().x(), center().y(), velocity().x(), velocity().y()};
    }
  }

  public int nOfInputsPerAgent() {
    return 1;
  }

  @Override
  public double[] defaultAgent1Action() {
    return new double[nOfInputsPerAgent()];
  }

  @Override
  public double[] defaultAgent2Action() {
    return new double[nOfInputsPerAgent()];
  }

  @Override
  public State getState() {
    return state;
  }

  //TODO check
  private BallState handleCollisionWithRacketIfAny(BallState updatedBallState, RacketState racketState) {
    double upperLimitRacketRectangle = configuration.racketsLength - configuration.racketsEdgeRadius;
    double lowerLimitRacketRectangle = -upperLimitRacketRectangle;
    Rectangle racketRectangle = new Rectangle(
        new Point(-configuration.racketsEdgeRadius, upperLimitRacketRectangle),
        new Point(configuration.racketsEdgeRadius, lowerLimitRacketRectangle)
    );
    Circle upperRacketCircle = new Circle(new Point(0, upperLimitRacketRectangle), configuration.racketsEdgeRadius);
    Circle lowerRacketCircle = new Circle(new Point(0, lowerLimitRacketRectangle), configuration.racketsEdgeRadius);
    BallState previousBallStateRRF = toRacketReferenceFrame(state.ballState, racketState); // RRF -> Racket Reference Frame
    BallState updatedBallStateRRF = toRacketReferenceFrame(updatedBallState, racketState);
    Segment racketTrajectoryRRF = new Segment(previousBallStateRRF.center, updatedBallStateRRF.center);
    List<Point> upperRacketCircleIntersections = upperRacketCircle.intersection(racketTrajectoryRRF);
    List<Point> lowerRacketCircleIntersections = lowerRacketCircle.intersection(racketTrajectoryRRF);
    List<Point> racketRectangleVerticalEdgesIntersections = racketRectangle.verticalEdgesIntersections(racketTrajectoryRRF);
    List<Point> allRacketIntersections = new ArrayList<>();
    allRacketIntersections.addAll(upperRacketCircleIntersections);
    allRacketIntersections.addAll(lowerRacketCircleIntersections);
    allRacketIntersections.addAll(racketRectangleVerticalEdgesIntersections);
    if (allRacketIntersections.isEmpty()) {
      return updatedBallState;
    }
    Point closestCollisionPoint = allRacketIntersections.getFirst();
    double minDistance = previousBallStateRRF.center.distance(closestCollisionPoint);
    allRacketIntersections.removeFirst();
    for (Point point : allRacketIntersections) {
      double distance = previousBallStateRRF.center.distance(point);
      if (distance < minDistance) {
        minDistance = distance;
        closestCollisionPoint = point;
      }
    }
    if (racketRectangleVerticalEdgesIntersections.contains(closestCollisionPoint)) {
      return toArenaReferenceFrame(
          racketVerticalEdgeCollisionWithSimplifiedSpinEffect(updatedBallState, racketState, closestCollisionPoint),
          racketState
      );
    } else if (upperRacketCircleIntersections.contains(closestCollisionPoint)) {
      return toArenaReferenceFrame(
          racketCircularEdgeCollisionWithSimplifiedSpinEffect(updatedBallState, racketState, closestCollisionPoint, upperRacketCircle.center()),
          racketState
      );
    } else {
      return toArenaReferenceFrame(
          racketCircularEdgeCollisionWithSimplifiedSpinEffect(updatedBallState, racketState, closestCollisionPoint, lowerRacketCircle.center()),
          racketState
      );
    }
  }

  //TODO check
  private BallState racketVerticalEdgeCollisionWithSimplifiedSpinEffect(BallState updatedBallState, RacketState racketState, Point collisionPoint) {
    Point mirroredReflection = new Point(collisionPoint.x() + (collisionPoint.x() - updatedBallState.center.x()), updatedBallState.center.y());
    Point mirroredVelocity = new Point(-updatedBallState.velocity().x(), updatedBallState.velocity().y());
    double correctionAngle = -(configuration.spinEffectFactor * racketState.yVelocity); //The minus sign is needed to simulate inverse proportionality that arises with the spin effect
    Point adjustedReflection = mirroredReflection.rotateCounterClockWise(collisionPoint, correctionAngle);
    Point adjustedVelocity = mirroredVelocity.rotateCounterClockWise(collisionPoint, correctionAngle);
    return new BallState(adjustedReflection, adjustedVelocity);
  }

  //TODO check
  private BallState racketCircularEdgeCollisionWithSimplifiedSpinEffect(BallState updatedBallState, RacketState racketState, Point collisionPoint, Point circularEdgeCenter) {
    double collisionPointRotationAngel = collisionPoint.getRotationAngleCounterClockwise(circularEdgeCenter);
    BallState ballStateRotatedVertically = updatedBallState.rotateCounterClockWise(circularEdgeCenter, collisionPointRotationAngel);
    Point collisionPointRotatedVertically = collisionPoint.rotateCounterClockWise(circularEdgeCenter, -collisionPointRotationAngel);
    BallState ballStateAfterVerticalCollision = racketVerticalEdgeCollisionWithSimplifiedSpinEffect(ballStateRotatedVertically, racketState, collisionPointRotatedVertically);
    return ballStateAfterVerticalCollision.rotateCounterClockWise(circularEdgeCenter, collisionPointRotationAngel);
  }

  // The racket reference frame is centered in the racket center with the x-axis pointing the center of the arena and the y-axis pointing upwards
  private BallState toRacketReferenceFrame(BallState ballState, RacketState racketState) {
    BallState ballStateFlippedReferenceFrame = ballState.deepCopy();
    if (racketState.side.equals(Side.RIGHT)) {
      ballStateFlippedReferenceFrame = flippedHorizontalAxisReferenceFrame(ballState);
    }
    return new BallState(
        new Point(
            ballStateFlippedReferenceFrame.center.x() - configuration.racketsEdgeRadius,
            ballStateFlippedReferenceFrame.center.y() - racketState.yCenter
        ),
        ballStateFlippedReferenceFrame.velocity
    );
  }

  private BallState toArenaReferenceFrame(BallState ballState, RacketState racketState) {
    BallState ballStateFlippedReferenceFrame = ballState.deepCopy();
    if (racketState.side.equals(Side.RIGHT)) {
      ballStateFlippedReferenceFrame = flippedHorizontalAxisReferenceFrame(ballState);
    }
    return new BallState(
        new Point(
            ballStateFlippedReferenceFrame.center.x() + configuration.racketsEdgeRadius,
            ballStateFlippedReferenceFrame.center.y() + racketState.yCenter
        ),
        ballStateFlippedReferenceFrame.velocity
    );
  }

  private BallState flippedHorizontalAxisReferenceFrame(BallState ballState) {
    return new BallState(
        new Point(
            configuration.arenaXLength() - ballState.center.x(),
            ballState.center.y()
        ),
        new Point(
            -ballState.velocity.x(),
            ballState.velocity.y()
        )
    );
  }

  private void checkPointEnd(BallState updatedBallState, double t) {
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
  }

  private BallState handleArenaEdgeMirroredVerticalCollisionIfAny(BallState ballState) {
    double offsetAbove = configuration.arenaYLength - ballState.center.y();
    double offsetBelow = ballState.center.y();
    if (offsetBelow * offsetAbove > 0.0d) {
      return ballState;
    }
    Point updatedBallVelocity = new Point(ballState.velocity().x(), -ballState.velocity().y());
    Point updatedBallPosition;
    if (offsetAbove < offsetBelow) {
      updatedBallPosition = new Point(ballState.center.x(), configuration.arenaYLength + offsetAbove);
    } else {
      updatedBallPosition = new Point(ballState.center.x(), -offsetBelow);
    }
    return new BallState(updatedBallPosition, updatedBallVelocity);
  }

  private BallState handleCollisionWithRacketsIfAny(BallState updatedBallState) {
    BallState processedBallStateCollisionLeftRacket = handleCollisionWithRacketIfAny(updatedBallState, state.lRacketState);
    if (processedBallStateCollisionLeftRacket.equals(updatedBallState)) {
      return handleCollisionWithRacketIfAny(updatedBallState, state.rRacketState);
    } else {
      return processedBallStateCollisionLeftRacket;
    }
  }

  @Override
  public void reset() {
    state = new State(
        configuration,
        new RacketState(
            configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
            0,
            Side.LEFT
        ),
        new RacketState(
            configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
            0,
            Side.RIGHT
        ),
        new BallState(
            new Point(0.0d, 0.0d),
            new Point(
                configuration.ballInitialXVelocityRange.denormalize(configuration.randomGenerator.nextDouble()),
                configuration.ballInitialXVelocityRange.denormalize(configuration.randomGenerator.nextDouble())
            )
        ),
        0,
        0
    );
    previousTime = 0d;
    pointStatus = PointStatus.IN_GAME;
    arena = new Rectangle(new Point(0.0, configuration.arenaYLength), new Point(configuration.arenaXLength, 0.0));
  }

  private void resetPointIfEnded() {
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
      double ballAccelerationPerDirection = Math.pow(configuration.ballAccelerationRatePerPoint, (lRacketScore + rRacketScore) / 2.0);
      state = new State(
          configuration,
          new RacketState(
              configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
              0,
              Side.LEFT
          ),
          new RacketState(
              configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
              0,
              Side.RIGHT
          ),
          new BallState(
              new Point(0.0d, 0.0d),
              new Point(
                  configuration.ballInitialXVelocityRange.denormalize(configuration.randomGenerator.nextDouble()) * ballAccelerationPerDirection,
                  configuration.ballInitialXVelocityRange.denormalize(configuration.randomGenerator.nextDouble()) * ballAccelerationPerDirection
              )
          ),
          lRacketScore,
          rRacketScore
      );
      previousTime = 0d;
      pointStatus = PointStatus.IN_GAME;
    }
  }

  private RacketState updateRacketState(RacketState racketState, double action, double deltaTime) {
    double racketDeltaYVelocity = DoubleRange.SYMMETRIC_UNIT.clip(action) * configuration.racketsMaxYVelocity;
    double updatedRacketYVelocity = racketState.yVelocity + racketDeltaYVelocity
        - deltaTime * configuration.racketsFriction * racketState.yVelocity;
    double updatedRacketY = racketState.yCenter + updatedRacketYVelocity * deltaTime;
    RacketState updatedRacketState = racketState.update(updatedRacketY, updatedRacketYVelocity);
    double upperEdgeArenaOffset = configuration.arenaYLength - (updatedRacketState.yCenter + configuration.racketsFriction / 2);
    double lowerEdgeArenaOffset = updatedRacketState.yCenter - configuration.racketsFriction / 2;
    if (upperEdgeArenaOffset * lowerEdgeArenaOffset > 0.0) {
      return updatedRacketState;
    } else if (upperEdgeArenaOffset < lowerEdgeArenaOffset) {
      return updatedRacketState.update(updatedRacketState.yCenter + upperEdgeArenaOffset, -updatedRacketState.yVelocity);
    } else {
      return updatedRacketState.update(updatedRacketState.yCenter - lowerEdgeArenaOffset, -updatedRacketState.yVelocity);
    }
  }

  private BallState updateBallState(BallState ballState, double deltaTime) {
    double deltaX = ballState.velocity.x() * deltaTime;
    double deltaY = ballState.velocity.y() * deltaTime;
    Point deltaPosition = new Point(deltaX, deltaY);
    return new BallState(ballState.center.sum(deltaPosition), ballState.velocity);
  }

  private void updateState(BallState ballState, RacketState lRacketState, RacketState rRacketState) {
    state = new State(
        configuration,
        lRacketState,
        rRacketState,
        ballState,
        state.lRacketScore,
        state.rRacketScore
    );
  }

  @Override
  public Pair<double[], double[]> step(double t, Pair<double[], double[]> actions) {
    resetPointIfEnded();
    if (actions.first().length != nOfInputsPerAgent()) {
      throw new IllegalArgumentException("Left agent action has wrong number of elements: %d found, %d expected"
          .formatted(actions.first().length, nOfInputsPerAgent()));
    }
    if (actions.second().length != nOfInputsPerAgent()) {
      throw new IllegalArgumentException("Right agent action has wrong number of elements: %d found, %d expected"
          .formatted(actions.second().length, nOfInputsPerAgent()));
    }
    // prepare
    double deltaTime = t - previousTime;
    previousTime = t;
    RacketState updatedLRacketState = updateRacketState(state.lRacketState, actions.first()[0], deltaTime);
    RacketState updatedRRacketState = updateRacketState(state.rRacketState, actions.second()[0], deltaTime);
    BallState updatedBallState = updateBallState(state.ballState, deltaTime);
    BallState updatedBallStateWithCollision = updatedBallState.deepCopy();
    checkPointEnd(updatedBallStateWithCollision, t);
    // boolean inGame = pointStatus == PointStatus.IN_GAME;
    // boolean collisionIsPossible = true;
    //while (collisionIsPossible && inGame) {
    //  updatedBallStateWithCollision = handleArenaEdgeMirroredVerticalCollisionIfAny(updatedBallStateWithCollision);
    //  updatedBallStateWithCollision = handleCollisionWithRacketsIfAny(updatedBallStateWithCollision);
    //  checkPointEnd(updatedBallStateWithCollision, t);
    //  collisionIsPossible = !updatedBallStateWithCollision.equals(updatedBallState);
    //  inGame = pointStatus == PointStatus.IN_GAME;
    //  updatedBallState = updatedBallStateWithCollision.deepCopy();
    //}
    updatedBallStateWithCollision = handleArenaEdgeMirroredVerticalCollisionIfAny(updatedBallStateWithCollision);
    updatedBallStateWithCollision = handleCollisionWithRacketsIfAny(updatedBallStateWithCollision);
    checkPointEnd(updatedBallStateWithCollision, t);
    updatedBallStateWithCollision = handleArenaEdgeMirroredVerticalCollisionIfAny(updatedBallStateWithCollision);
    updatedBallStateWithCollision = handleCollisionWithRacketsIfAny(updatedBallStateWithCollision);
    checkPointEnd(updatedBallStateWithCollision, t);

    updateState(updatedBallStateWithCollision, updatedLRacketState, updatedRRacketState);
    double[] lRacketObservation = DoubleStream.concat(
        Arrays.stream(updatedLRacketState.toArray()),
        Arrays.stream(updatedBallStateWithCollision.toArray())
    ).toArray();
    double[] rRacketObservation = DoubleStream.concat(
        Arrays.stream(updatedRRacketState.toArray()),
        Arrays.stream(flippedHorizontalAxisReferenceFrame(updatedBallStateWithCollision).toArray())
    ).toArray();
    return new Pair<>(lRacketObservation, rRacketObservation);
  }
}
