package io.github.ericmedvet.jsdynsym.core.numerical;

public class LinearAlgebraUtils {

  private LinearAlgebraUtils() {
  }

  public static double dotProduct(double[] v1, double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException("Vectors must have the same length for dot product.");
    }
    double sum = 0d;
    for (int i = 0; i < v1.length; i++) {
      sum += v1[i] * v2[i];
    }
    return sum;
  }

  public static double[] product(double[][] m, double[] v) {
    double[] o = new double[m.length];
    for (int j = 0; j < o.length; j++) {
      o[j] = dotProduct(m[j], v);
    }
    return o;
  }

}
